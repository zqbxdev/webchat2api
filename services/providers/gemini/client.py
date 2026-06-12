from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from curl_cffi import requests
from fastapi import HTTPException

from services.providers.base import ModelSpec
from services.providers.gemini.accounts import (
    cookie_header_from_mapping,
    cookie_header_from_response,
    gemini_cookie_state,
    gemini_rotate_cookies_result,
    gemini_session_token,
    merge_cookie_headers,
    merge_response_cookies,
    parse_cookie_header,
    sanitize_cookie_header,
)
from services.providers.gemini.models import gemini_model_metadata
from services.network.client import create_session

GEMINI_WEB_BASE_URL = "https://gemini.google.com"
GEMINI_GOOGLE_BASE_URL = "https://www.google.com"
GEMINI_ROTATE_COOKIES_URL = "https://accounts.google.com/RotateCookies"
GEMINI_WEB_GENERATE_URL = f"{GEMINI_WEB_BASE_URL}/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
GEMINI_ROTATE_COOKIES_BODY = '[000,"-0000000000000000000"]'
GEMINI_BROWSER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
GEMINI_GENERATE_MAX_ATTEMPTS = 3
GEMINI_RETRY_BACKOFF_SECONDS = 0.25
GEMINI_WEB_RPC_ID = "assistant.lamda.BardFrontendService.StreamGenerate"
GEMINI_WEB_IMAGE_UNSUPPORTED_DETAIL = "Gemini Web image input is not supported by this upstream adapter"
GEMINI_IMAGE_PART_TYPES = {"image", "image_url", "input_image"}
GEMINI_IMAGE_PAYLOAD_KEYS = {"image_url", "inlineData", "inline_data"}


@dataclass(frozen=True)
class GeminiCompletion:
    content: str
    raw_response: object = None
    metadata: dict[str, str] = field(default_factory=dict)


class GeminiWebError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502, upstream_status: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.upstream_status = upstream_status
        self.code = code

    def to_http_detail(self) -> dict[str, Any]:
        detail: dict[str, Any] = {"error": str(self)}
        if self.upstream_status is not None:
            detail["upstream_status"] = self.upstream_status
        if self.code:
            detail["code"] = self.code
        return detail


def session_token_from_response(raw_text: str) -> str:
    for pattern in (
        r'"SNlM0e"\s*:\s*"([^"]+)"',
        r'\[\s*"SNlM0e"\s*,\s*"([^"]+)"\s*\]',
        r'\[\s*"at"\s*,\s*"([^"]+)"\s*\]',
        r'\bat\s*[:=]\s*"([^"]+)"',
        r'\bnonce\s*=\s*"([^"]+)"',
    ):
        match = re.search(pattern, raw_text)
        if match:
            return match.group(1)
    return ""


def classify_upstream_error(status_code: int, raw_text: str = "") -> GeminiWebError:
    text = raw_text.lower()
    if status_code in {401, 403} or any(marker in text for marker in ("snlm0e", "secure-1psid", "secure-1psidts", "sign in", "accounts.google.com")):
        message = "Gemini upstream authentication failed"
        if "snlm0e" in text or "secure-1psidts" in text or "secure-1psid" in text:
            message = "Gemini upstream authentication failed; refresh Gemini session cookies"
        return GeminiWebError(message, status_code=401, upstream_status=status_code, code="gemini_auth_failed")
    if status_code == 429:
        return GeminiWebError("Gemini upstream rate limit exceeded", status_code=429, upstream_status=status_code, code="gemini_rate_limited")
    if status_code >= 500:
        return GeminiWebError("Gemini upstream service unavailable", status_code=502, upstream_status=status_code, code="gemini_upstream_unavailable")
    return GeminiWebError("Gemini upstream request failed", status_code=502, upstream_status=status_code, code="gemini_upstream_error")


def build_stream_generate_form_payload(prompt: str, model: str, session_token: str = "") -> dict[str, str]:
    inner = [[prompt], None, None, model]
    outer = [None, json.dumps(inner, ensure_ascii=False, separators=(",", ":"))]
    data = {"f.req": json.dumps(outer, ensure_ascii=False, separators=(",", ":"))}
    if session_token:
        data["at"] = session_token
    return data


def stream_generate_url(session_token: str = "") -> str:
    if not session_token:
        return GEMINI_WEB_GENERATE_URL
    from urllib.parse import quote

    return f"{GEMINI_WEB_GENERATE_URL}?at={quote(session_token, safe='')}"


def account_session_token(account: dict[str, Any]) -> str:
    return gemini_session_token(account)


def account_cookie_header(account: dict[str, Any]) -> str:
    try:
        return gemini_cookie_state(account, require_session_cookies=True).cookie_header
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc


def gemini_session_writeback(account: dict[str, Any], cookie_header: str, session_token: str = "") -> dict[str, Any]:
    cookies = parse_cookie_header(cookie_header)
    updates: dict[str, Any] = {}
    if cookies:
        updates["cookies"] = cookies
    for name in ("__Secure-1PSID", "__Secure-1PSIDTS"):
        if cookies.get(name):
            updates[name] = cookies[name]
    token = session_token or gemini_session_token(account)
    if token:
        updates["session_token"] = token
        updates["SNlM0e"] = token
        updates["at"] = token
    return updates


def persist_gemini_session(account_service: object, access_token: str, account: dict[str, Any], cookie_header: str, session_token: str = "") -> None:
    update_account = getattr(account_service, "update_account", None)
    if not callable(update_account):
        return
    updates = gemini_session_writeback(account, cookie_header, session_token)
    if updates:
        update_account(access_token, updates, provider="gemini")


def rotate_psidts_cookie(session: object, cookie_header: str, user_agent: str | None = None) -> str:
    headers = {
        "content-type": "application/json",
        "cookie": cookie_header,
        "origin": "https://accounts.google.com",
        "user-agent": user_agent or GEMINI_BROWSER_USER_AGENT,
    }
    post = getattr(session, "post")
    response = post(GEMINI_ROTATE_COOKIES_URL, headers=headers, data=GEMINI_ROTATE_COOKIES_BODY, timeout=30)
    status_code = int(getattr(response, "status_code", getattr(response, "status", 0)) or 0)
    raw_text = str(getattr(response, "text", "") or "")
    if status_code in {401, 403}:
        raise classify_upstream_error(status_code, raw_text)
    if status_code >= 400:
        raise classify_upstream_error(status_code, raw_text)
    rotation = gemini_rotate_cookies_result(cookie_header, response)
    if not rotation.psidts:
        raise GeminiWebError("Gemini cookie rotation did not retain __Secure-1PSIDTS", status_code=401, upstream_status=status_code, code="gemini_session_cookie_missing")
    return rotation.cookie_header


def contains_image_content(value: object) -> bool:
    if isinstance(value, dict):
        block_type = str(value.get("type") or "").strip()
        if block_type in GEMINI_IMAGE_PART_TYPES:
            return True
        if any(key in value for key in GEMINI_IMAGE_PAYLOAD_KEYS):
            return True
        return any(contains_image_content(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_image_content(item) for item in value)
    return False


def raise_unsupported_image_input() -> None:
    raise HTTPException(status_code=400, detail={"error": GEMINI_WEB_IMAGE_UNSUPPORTED_DETAIL})


def contains_image_content(value: object) -> bool:
    if isinstance(value, dict):
        block_type = str(value.get("type") or "").strip()
        if block_type in GEMINI_IMAGE_PART_TYPES:
            return True
        if any(key in value for key in GEMINI_IMAGE_PAYLOAD_KEYS):
            return True
        return any(contains_image_content(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_image_content(item) for item in value)
    return False


def raise_unsupported_image_input() -> None:
    raise HTTPException(status_code=400, detail={"error": GEMINI_WEB_IMAGE_UNSUPPORTED_DETAIL})


def message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "").strip()
        if block_type in {"", "text", "input_text", "output_text"}:
            text = block.get("text") or block.get("input_text") or block.get("output_text")
            if text:
                parts.append(str(text))
    return "\n".join(part for part in parts if part)


def build_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        if contains_image_content(message.get("content")):
            raise_unsupported_image_input()
        role = str(message.get("role") or "user").strip().lower()
        text = message_text(message.get("content")).strip()
        if not text:
            continue
        label = "User" if role == "user" else "Assistant" if role == "assistant" else "System"
        parts.append(f"{label}: {text}")
    if not parts:
        raise HTTPException(status_code=400, detail={"error": "Gemini chat requires at least one text message"})
    return "\n\n".join(parts)


def build_web_payload(spec: ModelSpec, body: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = build_prompt(messages)
    payload: dict[str, Any] = {
        "model": spec.upstream_model or spec.id,
        "prompt": prompt,
    }
    for key in ("temperature", "top_p", "max_tokens"):
        if body.get(key) is not None:
            payload[key] = body[key]
    return payload


def _json_loads(value: str) -> object | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _parse_nested_json(value: object) -> object:
    current = value
    while isinstance(current, str):
        text = current.strip()
        if not text.startswith(("[", "{")):
            return current
        parsed = _json_loads(text)
        if parsed is None:
            return current
        current = parsed
    return current


def _wrb_payload_from_entry(entry: object) -> object | None:
    if not isinstance(entry, list) or len(entry) < 3:
        return None
    if entry[0] != "wrb.fr":
        return None
    payload = _parse_nested_json(entry[2])
    if not isinstance(payload, list):
        return None
    if entry[1] == GEMINI_WEB_RPC_ID or _canonical_stream_generate_text(payload):
        return payload
    return None


def _wrb_payloads(value: object) -> Iterator[object]:
    payload = _wrb_payload_from_entry(value)
    if payload is not None:
        yield payload
        return
    if isinstance(value, list):
        for item in value:
            yield from _wrb_payloads(item)


def _has_wrb_frame(value: object) -> bool:
    if isinstance(value, list):
        if len(value) >= 3 and value[0] == "wrb.fr" and isinstance(_parse_nested_json(value[2]), list):
            return True
        return any(_has_wrb_frame(item) for item in value)
    return False


def _is_incidental_stream_string(value: str) -> bool:
    text = value.strip()
    lowered = text.lower()
    if not text:
        return True
    if text.isdigit():
        return True
    if lowered.startswith(("rc_", "wrb.fr", "assistant.lamda.", "http://", "https://")):
        return True
    if lowered in {"snlm0e", "at"}:
        return True
    if lowered.startswith("gemini-"):
        return True
    if lowered in {"generic", "rpc-id", "request-id", "conversation-id", "response-id"}:
        return True
    if lowered.endswith("-id"):
        return True
    return False


def _canonical_stream_generate_text(payload: object) -> str:
    payload = _parse_nested_json(payload)
    if not isinstance(payload, list) or len(payload) <= 4:
        return ""
    candidates = payload[4]
    if not isinstance(candidates, list) or not candidates:
        return ""
    first_candidate = candidates[0]
    if not isinstance(first_candidate, list) or len(first_candidate) <= 1:
        return ""
    candidate_body = first_candidate[1]
    if isinstance(candidate_body, list) and candidate_body:
        direct_text = candidate_body[0]
        if isinstance(direct_text, str) and direct_text.strip() and not _is_incidental_stream_string(direct_text):
            return direct_text.strip()
        if isinstance(direct_text, list) and direct_text:
            nested_text = direct_text[0]
            if isinstance(nested_text, str) and nested_text.strip() and not _is_incidental_stream_string(nested_text):
                return nested_text.strip()
    return ""


def _response_candidates_from_stream_generate(value: object) -> Iterator[str]:
    value = _parse_nested_json(value)
    if isinstance(value, dict):
        for key in ("content", "text", "answer", "response"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                yield candidate.strip()
        for item in value.values():
            yield from _response_candidates_from_stream_generate(item)
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text and not text.startswith(("[", "{")):
                    if not _is_incidental_stream_string(text):
                        yield text
                else:
                    yield from _response_candidates_from_stream_generate(item)
            elif isinstance(item, (list, dict)):
                yield from _response_candidates_from_stream_generate(item)


def extract_stream_generate_text(payload: object) -> str:
    wrb_payloads = list(_wrb_payloads(payload))
    canonical_texts = [
        canonical_text
        for wrb_payload in wrb_payloads
        if (canonical_text := _canonical_stream_generate_text(wrb_payload))
    ]
    if canonical_texts:
        return max(canonical_texts, key=len).strip()
    for wrb_payload in wrb_payloads:
        candidates = [candidate for candidate in _response_candidates_from_stream_generate(wrb_payload) if candidate]
        if candidates:
            return max(candidates, key=len).strip()
    return ""


def extract_stream_generate_metadata(payload: object) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for wrb_payload in _wrb_payloads(payload):
        parsed = _parse_nested_json(wrb_payload)
        if not isinstance(parsed, list):
            continue
        if len(parsed) > 1:
            ids = parsed[1]
            if isinstance(ids, list):
                if len(ids) > 0 and isinstance(ids[0], str) and ids[0].strip():
                    metadata.setdefault("cid", ids[0].strip())
                if len(ids) > 1 and isinstance(ids[1], str) and ids[1].strip():
                    metadata.setdefault("rid", ids[1].strip())
            elif isinstance(ids, str) and ids.strip():
                metadata.setdefault("cid", ids.strip())
        if len(parsed) > 2 and isinstance(parsed[2], str) and parsed[2].strip():
            metadata.setdefault("rid", parsed[2].strip())
        if len(parsed) > 4 and isinstance(parsed[4], list) and parsed[4]:
            first_candidate = parsed[4][0]
            if isinstance(first_candidate, list) and first_candidate:
                choice_id = first_candidate[0]
                if isinstance(choice_id, str) and choice_id.strip():
                    metadata.setdefault("rcid", choice_id.strip())
        for item in parsed:
            if isinstance(item, list) and item:
                first = item[0]
                if isinstance(first, str) and first.startswith("rc_"):
                    metadata.setdefault("rcid", first)
    return metadata


def _string_candidates(value: object) -> Iterator[str]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "{")):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                yield from _string_candidates(parsed)
                return
        if text:
            yield text
        return
    if isinstance(value, dict):
        for key in ("content", "text", "output", "answer", "response"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                yield item
        for item in value.values():
            yield from _string_candidates(item)
        return
    if isinstance(value, list):
        for item in reversed(value):
            yield from _string_candidates(item)


def extract_text(payload: object) -> str:
    targeted = extract_stream_generate_text(payload)
    if targeted:
        return targeted
    if any(True for _ in _wrb_payloads(payload)) or _has_wrb_frame(payload):
        return ""
    if isinstance(payload, dict):
        for key in ("content", "text", "output", "answer", "response"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for text in _string_candidates(payload):
        stripped = text.strip()
        if stripped:
            return stripped
    return ""


def parse_web_response_text(raw_text: str) -> object:
    text = raw_text.strip()
    if not text:
        return {}
    parsed = _json_loads(text)
    if parsed is not None:
        return parsed
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line or line.startswith(")]}'"):
            continue
        parsed = _json_loads(line)
        if parsed is not None:
            return parsed
    matches = re.findall(r"\[[\s\S]*\]|\{[\s\S]*\}", text)
    for candidate in reversed(matches):
        parsed = _json_loads(candidate)
        if parsed is not None:
            return parsed
    return text


class GeminiWebClient:
    def __init__(self, cookie_header: str, user_agent: str | None = None, account: dict[str, Any] | None = None) -> None:
        self.cookie_header = cookie_header
        self.user_agent = user_agent or GEMINI_BROWSER_USER_AGENT
        self.session_token = ""
        self.account = account if isinstance(account, dict) else None
        self.session = create_session(account=self.account)

    def __enter__(self) -> "GeminiWebClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        close = getattr(self.session, "close", None)
        if callable(close):
            close()

    def fetch_init_body(self) -> str:
        google_headers = {
            "user-agent": self.user_agent,
        }
        try:
            google_response = self.session.get(f"{GEMINI_GOOGLE_BASE_URL}/", headers=google_headers, timeout=30)
        except Exception:
            google_response = None
        if google_response is not None:
            self.cookie_header = merge_response_cookies(self.cookie_header, google_response)
        headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "max-age=0",
            "cookie": self.cookie_header,
            "origin": GEMINI_WEB_BASE_URL,
            "referer": f"{GEMINI_WEB_BASE_URL}/",
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate",
            "sec-fetch-site": "same-origin",
            "upgrade-insecure-requests": "1",
            "user-agent": self.user_agent,
            "x-same-domain": "1",
        }
        response = self.session.get(f"{GEMINI_WEB_BASE_URL}/app?hl=en", headers=headers, timeout=30)
        status_code = int(getattr(response, "status_code", getattr(response, "status", 0)) or 0)
        raw_text = str(getattr(response, "text", "") or "")
        self.cookie_header = merge_response_cookies(self.cookie_header, response)
        if status_code >= 400:
            raise classify_upstream_error(status_code, raw_text)
        return raw_text

    def bootstrap_session_token(self) -> str:
        raw_text = self.fetch_init_body()
        session_token = session_token_from_response(raw_text)
        if session_token:
            self.session_token = session_token
            return session_token
        try:
            self.rotate_psidts()
            raw_text = self.fetch_init_body()
            session_token = session_token_from_response(raw_text)
            if session_token:
                self.session_token = session_token
                return session_token
        except GeminiWebError:
            raise
        except Exception:
            pass
        if "signin" in raw_text.lower() or "sign in" in raw_text.lower() or "accounts.google.com" in raw_text.lower():
            raise GeminiWebError("Gemini upstream authentication failed", status_code=401, upstream_status=None, code="gemini_auth_failed")
        raise GeminiWebError("Gemini session token bootstrap failed", status_code=401, upstream_status=None, code="gemini_session_token_missing")

    def rotate_psidts(self) -> str:
        self.cookie_header = rotate_psidts_cookie(self.session, self.cookie_header, self.user_agent)
        return self.cookie_header

    def _request_generate_once(self, payload: dict[str, Any], session_token: str) -> object:
        headers = {
            "content-type": "application/x-www-form-urlencoded;charset=UTF-8",
            "cookie": self.cookie_header,
            "user-agent": self.user_agent,
            "origin": GEMINI_WEB_BASE_URL,
            "referer": f"{GEMINI_WEB_BASE_URL}/",
            "x-same-domain": "1",
        }
        response = self.session.post(
            stream_generate_url(session_token),
            headers=headers,
            data=build_stream_generate_form_payload(payload["prompt"], payload["model"], session_token),
            timeout=120,
        )
        status_code = int(getattr(response, "status_code", getattr(response, "status", 0)) or 0)
        raw_text = str(getattr(response, "text", "") or "")
        if status_code >= 400:
            raise classify_upstream_error(status_code, raw_text)
        parsed = parse_web_response_text(raw_text)
        if not raw_text.strip() or not extract_text(parsed):
            raise GeminiWebError("Gemini upstream response did not contain text", status_code=502, upstream_status=status_code, code="gemini_empty_response")
        self.cookie_header = merge_response_cookies(self.cookie_header, response)
        return parsed

    def generate(self, payload: dict[str, Any]) -> object:
        session_token = str(payload.get("session_token") or "").strip()
        self.session_token = session_token
        if not session_token:
            session_token = self.bootstrap_session_token()
        last_error: GeminiWebError | Exception | None = None
        for attempt in range(GEMINI_GENERATE_MAX_ATTEMPTS):
            if attempt:
                time.sleep(GEMINI_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            try:
                return self._request_generate_once(payload, session_token)
            except GeminiWebError as exc:
                last_error = exc
                if exc.code in {"gemini_auth_failed", "gemini_rate_limited"}:
                    raise
                if exc.upstream_status is not None and exc.upstream_status < 500 and exc.code != "gemini_empty_response":
                    raise
            except Exception as exc:
                last_error = exc
        if isinstance(last_error, GeminiWebError):
            raise last_error
        raise GeminiWebError("Gemini upstream request failed", status_code=502, code="gemini_upstream_error") from last_error


def fetch_authenticated_init_body() -> str:
    from services.account_service import account_service

    access_token = account_service.get_text_access_token(provider="gemini")
    if not access_token:
        return ""
    account = account_service.get_account(access_token) or {"access_token": access_token, "provider": "gemini"}
    cookie_header = account_cookie_header(account)
    with GeminiWebClient(cookie_header, account.get("user_agent"), account=account) as client:
        init_body = client.fetch_init_body()
        persist_gemini_session(account_service, access_token, account, client.cookie_header)
        return init_body


def list_model_metadata() -> list[dict[str, Any]]:
    return [dict(item) for item in gemini_model_metadata()]


def extract_completion(payload: object) -> GeminiCompletion:
    return GeminiCompletion(content=extract_text(payload), raw_response=payload, metadata=extract_stream_generate_metadata(payload))


def chat_completion(body: dict[str, Any], spec: ModelSpec, messages: list[dict[str, Any]]) -> GeminiCompletion:
    from services.account_service import account_service

    payload = build_web_payload(spec, body, messages)
    access_token = account_service.get_text_access_token(provider="gemini")
    if not access_token:
        raise HTTPException(status_code=503, detail={"error": "no available Gemini account"})
    account = account_service.get_account(access_token) or {"access_token": access_token, "provider": "gemini"}
    cookie_header = account_cookie_header(account)
    session_token = account_session_token(account)
    if session_token:
        payload["session_token"] = session_token
    try:
        with GeminiWebClient(cookie_header, account.get("user_agent"), account=account) as client:
            response_payload = client.generate(payload)
            persist_gemini_session(account_service, access_token, account, client.cookie_header, client.session_token)
    except GeminiWebError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_http_detail()) from exc
    completion = extract_completion(response_payload)
    if not completion.content:
        raise HTTPException(status_code=502, detail={"error": "Gemini upstream response did not contain text"})
    account_service.mark_text_used(access_token)
    return completion


def synthetic_stream_content(content: str, chunk_size: int = 120) -> Iterator[str]:
    if not content:
        yield ""
        return
    for index in range(0, len(content), chunk_size):
        yield content[index:index + chunk_size]
