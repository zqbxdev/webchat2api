from __future__ import annotations

import base64
import json
import re
import threading
import time
import uuid
from urllib.parse import urlparse
from ipaddress import ip_address
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, cast

from curl_cffi import requests  # type: ignore[import-not-found]
from fastapi import HTTPException  # type: ignore[import-not-found]

from services.config import config
from services.providers.base import GROK_PROVIDER, ModelSpec
from services.providers.grok.models import GROK_MODEL_SPECS, is_supported_grok_app_chat_image_model
from services.network.client import create_session
from services.network.flaresolverr import FlareSolverrClearanceProvider
from services.network.headers import build_grok_console_headers
from services.network.profiles import build_grok_app_chat_profile, build_grok_console_profile, infer_chromium_impersonate
from services.providers.base import ImageGenerationError, ImageOutput
from utils.log import logger

CONSOLE_BASE_URL = "https://console.x.ai"
CONSOLE_RESPONSES_URL = f"{CONSOLE_BASE_URL}/v1/responses"
APP_CHAT_BASE_URL = "https://grok.com"
APP_CHAT_NEW_CONVERSATION_URL = f"{APP_CHAT_BASE_URL}/rest/app-chat/conversations/new"
APP_CHAT_RATE_LIMITS_URL = f"{APP_CHAT_BASE_URL}/rest/rate-limits"
APP_CHAT_UPLOAD_FILE_URL = f"{APP_CHAT_BASE_URL}/rest/app-chat/upload-file"
APP_CHAT_MEDIA_POST_CREATE_URL = f"{APP_CHAT_BASE_URL}/rest/media/post/create"
GROK_ASSET_BASE_URL = "https://assets.grok.com/"
GROK_APP_CHAT_STATSIG_ID = "0196a8f6-0501-79f8-8d74-a2f2c0f5f5f5"
GROK_IMAGE_EDIT_MODEL_NAME = "imagine-image-edit"
GROK_IMAGE_EDIT_MODEL_KIND = "imagine"
GROK_IMAGE_EDIT_MEDIA_TYPE = "MEDIA_POST_TYPE_IMAGE"
GROK_IMAGE_EDIT_SIZE = "1024x1024"
GROK_IMAGE_EDIT_MAX_REFERENCES = 7
GROK_IMAGE_EDIT_MAX_N = 2
GROK_IMAGE_ASPECT_RATIOS = {
    "1024x1024": "1:1",
    "1280x720": "16:9",
    "720x1280": "9:16",
    "1792x1024": "3:2",
    "1024x1792": "2:3",
    "1:1": "1:1",
    "2:3": "2:3",
    "3:2": "3:2",
    "9:16": "9:16",
    "16:9": "16:9",
}
GROK_IMAGE_RESPONSE_FORMATS = {"url", "b64_json", "base64"}
GROK_IMAGE_DOWNLOAD_TIMEOUT = 30
SEARCH_SOURCES_MARKER = "[webchat2api-sources]: #"
_GROK_IMAGE_PLACEHOLDER_RE = re.compile(r"@IMAGE(\d+)\b", re.IGNORECASE)
_SEARCH_SOURCES_BLOCK_RE = re.compile(r"\n{0,2}\[webchat2api-sources\]: #\n+## Sources\n(?:\d+\. \[[^\n\]]*\]\([^\n)]*\)\n?)+\s*", re.MULTILINE)
_APP_CHAT_AUTH_ERROR_MARKERS = (
    "invalid-credentials",
    "invalid_credentials",
    "bad-credentials",
    "bad_credentials",
    "bad credentials",
    "failed to look up session id",
    "blocked-user",
    "blocked_user",
    "email-domain-rejected",
    "email_domain_rejected",
    "session not found",
    "account suspended",
    "token revoked",
    "token_revoked",
    "token expired",
    "token_expired",
    "token-expired",
)
_APP_CHAT_RATE_LIMIT_MARKERS = (
    "rate_limit_exceeded",
    "rate-limit-exceeded",
    "rate limit",
    "rate-limit",
    "rate limited",
    "rate_limited",
    "too many requests",
    "quota exhausted",
    "quota_exhausted",
)
_APP_CHAT_TRANSIENT_ERROR_MARKERS = (
    "timed out",
    "timeout",
    "connection reset",
    "temporarily unavailable",
    "http2",
    "http/2",
    "stream reset",
    "upstream connect error",
)
_APP_CHAT_CHALLENGE_MARKERS = (
    "cloudflare",
    "cf-chl",
    "cf_clearance",
    "cf-mitigated",
    "turnstile",
    "challenge-platform",
    "checking your browser",
    "just a moment",
    "attention required",
    "enable javascript and cookies",
    "captcha",
    "interstitial",
    "browser integrity check",
)
_APP_CHAT_AUTH_STATUS_CODES = {401, 403}
_APP_CHAT_LIMIT_STATUS_CODES = {402, 429}
_APP_CHAT_TRANSIENT_STATUS_CODES = {408, 500, 502, 503, 504}
_APP_CHAT_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_APP_CHAT_MAX_ACCOUNT_ATTEMPTS = 3
_APP_CHAT_PUBLIC_ERROR_PREFIX = "Grok app-chat upstream error"
_APP_CHAT_CLEARANCE_LOCK = threading.Lock()

GROK_RATE_LIMIT_MODEL_NAME = "auto"
GROK_SUPER_WINDOW_THRESHOLD_SECONDS = 14400


def _int_or_none(value: object) -> int | None:
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _walk_dict_values(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_dict_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_dict_values(item)


def _rate_limit_model_name(account: dict[str, Any] | None = None) -> str:
    if isinstance(account, dict):
        for key in ("modelName", "model_name", "upstream_model", "model", "model_id"):
            value = str(account.get(key) or "").strip()
            if value:
                return value
    return GROK_RATE_LIMIT_MODEL_NAME


def build_grok_rate_limits_payload(account: dict[str, Any] | None = None) -> dict[str, str]:
    return {"modelName": _rate_limit_model_name(account)}


def extract_grok_rate_limit_value(payload: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for item in _walk_dict_values(payload):
        for key in keys:
            if key in item:
                value = _int_or_none(item.get(key))
                if value is not None:
                    return value
    return None


def extract_grok_rate_limit_remaining(payload: dict[str, Any]) -> int | None:
    return extract_grok_rate_limit_value(payload, ("remainingQueries", "remaining_queries", "remaining_tokens", "remainingTokens"))


def extract_grok_rate_limit_total(payload: dict[str, Any]) -> int | None:
    return extract_grok_rate_limit_value(payload, ("totalQueries", "total_queries", "totalTokens", "total_tokens"))


def extract_grok_rate_limit_window_seconds(payload: dict[str, Any]) -> int | None:
    return extract_grok_rate_limit_value(payload, ("windowSizeSeconds", "window_size_seconds"))


def grok_rate_limit_account_hints(payload: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {"app_chat": True}
    remaining = extract_grok_rate_limit_remaining(payload)
    if remaining is not None:
        updates["quota"] = max(0, remaining)
        updates["status"] = "正常" if remaining > 0 else "限流"
    total = extract_grok_rate_limit_total(payload)
    if total is not None:
        updates["quota_total"] = max(0, total)
    window_seconds = extract_grok_rate_limit_window_seconds(payload)
    if window_seconds is not None:
        updates["rate_limit_window_seconds"] = window_seconds
        updates["tier"] = "basic" if window_seconds >= GROK_SUPER_WINDOW_THRESHOLD_SECONDS else "super"
    explicit_tier = ""
    for item in _walk_dict_values(payload):
        for key in ("tier", "modelTier", "model_tier", "subscriptionTier", "subscription_tier"):
            value = str(item.get(key) or "").strip().lower()
            if value in {"basic", "free", "fast", "super", "premium", "supergrok", "super-grok", "heavy", "max"}:
                from services.providers.grok.accounts import normalize_tier

                explicit_tier = normalize_tier(value)
                break
        if explicit_tier:
            break
    if explicit_tier:
        updates["tier"] = explicit_tier
    return updates

_BRIDGE_DEFAULT_URL = "http://127.0.0.1:3080"
_bridge_detected_url: str | None = None
_bridge_probed = False


class GrokConsoleError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        upstream_status: int | None = None,
        code: str | None = None,
        extra_detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.upstream_status = upstream_status
        self.code = code
        self.extra_detail = extra_detail or {}

    @property
    def is_check_unavailable(self) -> bool:
        return self.code in {
            "cloudflare_challenge",
            "invalid_rate_limit_response",
            "rate_limit_check_unavailable",
            "rate_limit_network_error",
            "upstream_transient",
        }

    def to_http_detail(self) -> dict[str, Any]:
        detail: dict[str, Any] = {"error": str(self)}
        if self.code:
            detail["code"] = self.code
        detail.update(self.extra_detail)
        return detail


@dataclass(frozen=True)
class GrokConsoleCompletion:
    content: str
    reasoning_content: str = ""
    raw_reasoning: object = None
    raw_usage: object = None
    raw_response: dict[str, Any] | None = None


@dataclass(frozen=True)
class GrokConsoleStreamDelta:
    content: str = ""
    reasoning_content: str = ""


@dataclass(frozen=True)
class GrokImageEditReference:
    file_id: str
    content_url: str


_THINKING_SUMMARY_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:思考摘要|思考总结|thinking\s+summary|thought\s+summary|reasoning\s+summary|thinking|reasoning)\s*(?:\*\*\s*[:：]|[:：]\s*(?:\*\*)?)\s*(.*)$",
    re.IGNORECASE,
)
_ANSWER_SUMMARY_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(?:答案|回答|answer|final\s+answer|response)\s*(?:\*\*\s*[:：]|[:：]\s*(?:\*\*)?)\s*(.*)$",
    re.IGNORECASE,
)
_CONSOLE_SEARCH_TOOL_TYPES = {"web_search", "x_search"}


def split_visible_console_reasoning(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    lines = text.splitlines(keepends=True)
    first_content_index = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first_content_index is None:
        return text, ""
    thinking_match = _THINKING_SUMMARY_RE.match(lines[first_content_index].rstrip("\r\n"))
    if not thinking_match:
        return text, ""
    answer_index = -1
    answer_match: re.Match[str] | None = None
    for index in range(first_content_index + 1, len(lines)):
        candidate = _ANSWER_SUMMARY_RE.match(lines[index].rstrip("\r\n"))
        if candidate:
            answer_index = index
            answer_match = candidate
            break
    if answer_index < 0 or answer_match is None:
        return text, ""

    reasoning_parts: list[str] = []
    if thinking_match.group(1):
        reasoning_parts.append(thinking_match.group(1))
        if lines[first_content_index].endswith(("\n", "\r")) and first_content_index + 1 < answer_index:
            reasoning_parts.append("\n")
    reasoning_parts.extend(lines[first_content_index + 1:answer_index])
    content_parts = lines[:first_content_index]
    if answer_match.group(1):
        content_parts.append(answer_match.group(1))
        if lines[answer_index].endswith(("\n", "\r")):
            content_parts.append("\n")
    content_parts.extend(lines[answer_index + 1:])
    return "".join(content_parts).strip(), "".join(reasoning_parts).strip()


def _message_text(content: object, role: str) -> str:
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
        if block_type in {"text", "input_text", "output_text"}:
            text = block.get("text") or block.get("input_text") or block.get("output_text")
            if text:
                parts.append(str(text))
    return "\n".join(part for part in parts if part)


def build_console_input(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().lower()
        text = _message_text(message.get("content"), role).strip()
        if not text:
            continue
        if role == "system":
            instructions.append(text)
            continue
        if role not in {"user", "assistant"}:
            continue
        content_type = "output_text" if role == "assistant" else "input_text"
        input_items.append({
            "role": role,
            "content": [{"type": content_type, "text": text}],
        })
    return "\n\n".join(instructions).strip(), input_items


def _console_search_tools(tools: object) -> list[dict[str, Any]]:
    search_tools: list[dict[str, Any]] = []
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if str(tool.get("type") or "") in _CONSOLE_SEARCH_TOOL_TYPES:
                search_tools.append(dict(tool))
    if not any(tool.get("type") == "web_search" for tool in search_tools):
        search_tools.append({"type": "web_search"})
    return search_tools


def build_console_payload(spec: ModelSpec, body: dict[str, Any], messages: list[dict[str, Any]]) -> dict[str, Any]:
    instructions, input_items = build_console_input(messages)
    if not input_items:
        raise HTTPException(status_code=400, detail={"error": "Grok chat requires at least one user or assistant text message"})
    payload: dict[str, Any] = {
        "model": spec.upstream_model or spec.id,
        "input": input_items,
        "tools": _console_search_tools(body.get("tools")),
    }
    request_instructions = str(body.get("instructions") or "").strip()
    merged_instructions = "\n\n".join(item for item in [request_instructions, instructions] if item)
    if merged_instructions:
        payload["instructions"] = merged_instructions
    for key in ("temperature", "top_p", "max_output_tokens", "max_tokens"):
        if body.get(key) is not None:
            target_key = "max_output_tokens" if key == "max_tokens" else key
            payload[target_key] = body[key]
    for key in ("tool_choice", "parallel_tool_calls"):
        if body.get(key) is not None:
            payload[key] = body[key]
    reasoning_effort = str(body.get("reasoning_effort") or spec.default_reasoning_effort or "").strip().lower()
    if reasoning_effort and reasoning_effort != "none":
        payload["reasoning"] = {"effort": "high" if reasoning_effort == "xhigh" else reasoning_effort}
    return payload


def _extract_text_from_content(content: object) -> str:
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
        text = block.get("text") or block.get("output_text") or block.get("input_text")
        if text and str(block.get("type") or "") in {"", "text", "output_text", "input_text"}:
            parts.append(str(text))
    return "".join(parts)


def extract_console_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    output = payload.get("output")
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type in {"message", "output_message", ""}:
            text = _extract_text_from_content(item.get("content"))
            if text:
                parts.append(text)
        elif item_type in {"output_text", "text"} and item.get("text"):
            parts.append(str(item.get("text")))
    return "".join(parts)


def extract_console_completion(payload: dict[str, Any]) -> GrokConsoleCompletion:
    text = extract_console_text(payload)
    content, reasoning_content = split_visible_console_reasoning(text)
    return GrokConsoleCompletion(
        content=content,
        reasoning_content=reasoning_content,
        raw_reasoning=payload.get("reasoning"),
        raw_usage=payload.get("usage"),
        raw_response=payload,
    )


def _text_field(value: object) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    for key in ("text", "content", "output_text", "reasoning_content", "summary_text"):
        text = value.get(key)
        if isinstance(text, str) and text:
            return text
    return ""


def extract_console_stream_delta(event: dict[str, Any]) -> GrokConsoleStreamDelta:
    event_type = str(event.get("type") or "").lower()
    if event_type and "delta" not in event_type:
        return GrokConsoleStreamDelta()
    text = _text_field(event.get("delta"))
    if not text:
        text = _text_field(event)
    if not text:
        return GrokConsoleStreamDelta()
    if "reasoning" in event_type or "thinking" in event_type:
        return GrokConsoleStreamDelta(reasoning_content=text)
    return GrokConsoleStreamDelta(content=text)


def _parse_console_stream_payload(payload: str, current_event: str) -> dict[str, Any] | None:
    if not payload:
        return None
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning({"event": "grok_console_stream_invalid_json"})
        return None
    if not isinstance(event, dict):
        return None
    if current_event and not event.get("type"):
        event = {"type": current_event, **event}
    return event


def _iter_console_stream_events(lines: Iterable[object]) -> Iterator[dict[str, Any]]:
    current_event = ""
    data_lines: list[str] = []

    def flush_data() -> dict[str, Any] | None:
        nonlocal current_event
        if not data_lines:
            current_event = ""
            return None
        payload = "\n".join(data_lines).strip()
        data_lines.clear()
        event = _parse_console_stream_payload(payload, current_event)
        current_event = ""
        return event

    for raw_line in lines:
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.rstrip("\r\n")
        if not line.strip():
            event = flush_data()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = flush_data()
            if event is not None:
                yield event
            current_event = line[6:]
            if current_event.startswith(" "):
                current_event = current_event[1:]
            current_event = current_event.strip()
            continue
        if line.startswith("data:"):
            payload = line[5:]
            if payload.startswith(" "):
                payload = payload[1:]
            if payload.strip() == "[DONE]":
                event = flush_data()
                if event is not None:
                    yield event
                break
            data_lines.append(payload)
            continue
        line = line.strip()
        if line.startswith("{"):
            event = flush_data()
            if event is not None:
                yield event
            event = _parse_console_stream_payload(line, current_event)
            if event is not None:
                yield event

    event = flush_data()
    if event is not None:
        yield event


def _raise_for_console_stream_event(event: dict[str, Any]) -> None:
    event_type = str(event.get("type") or "").lower()
    if event_type not in {"error", "response.failed", "response.error", "response.incomplete", "response.cancelled"}:
        return
    error = event.get("error")
    response = event.get("response")
    if not error and isinstance(response, dict):
        error = response.get("error") or response.get("incomplete_details")
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("code") or error.get("reason") or event_type)
    elif error:
        message = str(error)
    else:
        message = event_type
    raise GrokConsoleError(f"Grok upstream stream error: {message}", 502)


def _grok_console_profile():
    return build_grok_console_profile(config.data)


def _account_text(account: dict[str, Any] | None, *keys: str) -> str:
    if not isinstance(account, dict):
        return ""
    for key in keys:
        value = account.get(key)
        if isinstance(value, dict):
            continue
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _app_chat_profile_value(profile: object, account: dict[str, Any] | None, field: str, *account_keys: str) -> str:
    return _account_text(account, *account_keys) or str(getattr(profile, field, "") or "")


def _app_chat_impersonate(profile: object, account: dict[str, Any] | None) -> str:
    return _account_text(account, "impersonate", "browser") or str(getattr(profile, "impersonate", "") or "")


def _grok_app_chat_profile():
    return build_grok_app_chat_profile(config.data)


def _headers(access_token: str) -> dict[str, str]:
    return build_grok_console_headers(_grok_console_profile(), access_token=access_token, base_url=CONSOLE_BASE_URL)


def _openai_status(upstream_status: int) -> int:
    if upstream_status in {401, 403, 404}:
        return upstream_status
    if upstream_status in {402, 429}:
        return 429
    if 400 <= upstream_status < 500:
        return 400
    return 502


def _feedback_status(upstream_status: int) -> str | None:
    if upstream_status in {401, 403}:
        return "异常"
    if upstream_status in {402, 429}:
        return "限流"
    return None


def _app_chat_feedback_status(upstream_status: int) -> str | None:
    if upstream_status in {401, 403}:
        return "异常"
    if upstream_status in {402, 429}:
        return "限流"
    return None


def _console_upstream_error_detail(response: object | None) -> str:
    if response is None:
        return ""
    json_data: object = None
    json_method = getattr(response, "json", None)
    if callable(json_method):
        try:
            json_data = json_method()
        except Exception:
            json_data = None
    if isinstance(json_data, dict):
        error = json_data.get("error")
        if isinstance(error, dict):
            for key in ("message", "code", "reason", "type"):
                value = error.get(key)
                if value:
                    return str(value)
        elif error:
            return str(error)
        for key in ("message", "detail", "code", "reason"):
            value = json_data.get(key)
            if value:
                return str(value)
    text = getattr(response, "text", "") or ""
    if not text:
        content = getattr(response, "content", b"")
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
    return str(text).strip()[:400]


def _raise_console_upstream_error(access_token: str, upstream_status: int, response: object | None = None) -> None:
    feedback_status = _feedback_status(upstream_status)
    if feedback_status:
        from services.account_service import account_service

        account_service.update_account(access_token, {"status": feedback_status})
    message = f"Grok upstream error (HTTP {upstream_status})"
    detail = _console_upstream_error_detail(response)
    if detail:
        message = f"{message}: {detail}"
    raise GrokConsoleError(message, _openai_status(upstream_status), upstream_status)


class GrokConsoleClient:
    def __init__(self, access_token: str) -> None:
        self.access_token = access_token
        self.network_profile = _grok_console_profile()
        from services.account_service import account_service

        account = account_service.get_account(access_token, provider=GROK_PROVIDER)
        account = account if isinstance(account, dict) else None
        self.session = create_session(impersonate=self.network_profile.impersonate, verify=self.network_profile.verify, account=account)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "GrokConsoleClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _call_with_retry(self, fn, policy=None, context=""):
        """Wrap a callable with retry handling."""
        from services.network.retry import retry_call, RetryPolicy

        api_policy = policy or RetryPolicy(
            max_attempts=3,
            retry_statuses=frozenset({408, 429, 500, 502, 503, 504}),
        )
        deadline = time.monotonic() + 60.0

        def on_retry(attempt, status_code, exc):
            logger.warning({
                "event": "grok_retry",
                "context": context,
                "attempt": attempt,
                "status_code": status_code,
                "error": str(exc) if exc else None,
            })

        return retry_call(fn, policy=api_policy, deadline=deadline, on_retry=on_retry)

    def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._call_with_retry(
                lambda: self.session.post(
                    CONSOLE_RESPONSES_URL,
                    headers=_headers(self.access_token),
                    json=payload,
                    timeout=self.network_profile.timeout,
                ),
                context="create_response",
            )
        except requests.exceptions.RequestException as exc:
            raise GrokConsoleError(f"Grok upstream request failed: {exc}", 502) from exc
        if response.status_code >= 400:
            _raise_console_upstream_error(self.access_token, int(response.status_code), response)
        data = response.json()
        if not isinstance(data, dict):
            raise GrokConsoleError("Grok upstream returned an invalid response", 502)
        return data

    def stream_response(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        try:
            response = self._call_with_retry(
                lambda: self.session.post(
                    CONSOLE_RESPONSES_URL,
                    headers=_headers(self.access_token),
                    json=stream_payload,
                    timeout=self.network_profile.timeout,
                    stream=True,
                ),
                context="stream_response",
            )
        except requests.exceptions.RequestException as exc:
            raise GrokConsoleError(f"Grok upstream request failed: {exc}", 502) from exc
        if response.status_code >= 400:
            _raise_console_upstream_error(self.access_token, int(response.status_code), response)
        try:
            for event in _iter_console_stream_events(response.iter_lines()):
                _raise_for_console_stream_event(event)
                yield event
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()


def _cookie_items(cookie_header: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for fragment in str(cookie_header or "").split(";"):
        name, separator, value = fragment.strip().partition("=")
        clean_name = " ".join(name.strip().split()).lower()
        clean_value = " ".join(value.strip().split())
        if separator and clean_name:
            items.append((clean_name, clean_value))
    return items


def _set_cookie(cookies: list[tuple[str, str]], name: str, value: str) -> None:
    for index, (existing_name, _) in enumerate(cookies):
        if existing_name == name:
            cookies[index] = (name, value)
            return
    cookies.append((name, value))


def _app_chat_cookie(access_token: str, cf_clearance: str = "", cf_cookies: str = "") -> str:
    token = str(access_token or "").strip()
    cookies: list[tuple[str, str]] = []
    sso_value = ""
    for name, value in _cookie_items(token):
        if name == "sso":
            sso_value = value
            break
    if not sso_value and token and "=" not in token:
        sso_value = " ".join(token.split())
    if sso_value:
        cookies.extend((name, value) for name, value in _cookie_items(token) if name not in {"sso", "sso-rw"})
        cookies.insert(0, ("sso-rw", sso_value))
        cookies.insert(0, ("sso", sso_value))
    else:
        cookies.extend(_cookie_items(token))
    for name, value in _cookie_items(cf_cookies):
        if name not in {"sso", "sso-rw"}:
            _set_cookie(cookies, name, value)
    clearance = " ".join(str(cf_clearance or "").strip().split())
    if clearance:
        _set_cookie(cookies, "cf_clearance", clearance)
    return "; ".join(f"{name}={value}" for name, value in cookies)


def _extract_raw_sso(access_token: str) -> str:
    token = str(access_token or "").strip()
    for name, value in _cookie_items(token):
        if name == "sso":
            return value
    if token and "=" not in token:
        return " ".join(token.split())
    return token


def _is_loopback_bridge_url(value: str) -> bool:
    parsed = urlparse(str(value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return False
    host = parsed.hostname.lower()
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _detect_bridge_url() -> str:
    configured = config.browser_bridge_url
    if configured:
        bridge_url = str(configured).strip().rstrip("/")
        if not _is_loopback_bridge_url(bridge_url):
            raise GrokConsoleError(
                "Grok app-chat Browser Bridge URL must be loopback",
                400,
                None,
                code="grok_browser_bridge_url_not_loopback",
                extra_detail={"bridge_code": "bridge_url_not_loopback"},
            )
        return bridge_url
    global _bridge_detected_url, _bridge_probed
    if _bridge_probed:
        return _bridge_detected_url or ""
    _bridge_probed = True
    import urllib.request
    try:
        with urllib.request.urlopen(f"{_BRIDGE_DEFAULT_URL}/health", timeout=2) as r:
            if r.status == 200:
                _bridge_detected_url = _BRIDGE_DEFAULT_URL
                logger.info({"event": "browser_bridge_detected", "url": _BRIDGE_DEFAULT_URL})
                return _BRIDGE_DEFAULT_URL
    except Exception:
        pass
    _bridge_detected_url = ""
    return ""


_BRIDGE_HEALTH_TIMEOUT = 2
_BRIDGE_EXPLICIT_CHAT_TIMEOUT = 55
_BRIDGE_ERROR_STATUS = {
    "invalid_json": 400,
    "invalid_request": 400,
    "bridge_unavailable": 503,
    "browser_bridge_unavailable": 503,
    "navigation_timeout": 504,
    "sso_unavailable": 401,
    "page_not_prepared": 503,
    "page_busy": 429,
    "request_timeout": 504,
    "upstream_error": 502,
}


def _browser_bridge_error_code(code: object | None) -> str:
    normalized = str(code or "browser_bridge_unavailable").strip() or "browser_bridge_unavailable"
    if normalized == "bridge_unavailable":
        normalized = "browser_bridge_unavailable"
    return normalized if normalized.startswith("grok_") else f"grok_{normalized}"


def _parse_bridge_json_response(response: object) -> dict[str, Any]:
    reader = getattr(response, "read", None)
    if not callable(reader):
        return {}
    try:
        raw = reader()
    except Exception:
        return {}
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw or "")
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _safe_bridge_message(bridge_code: str) -> str:
    if bridge_code == "navigation_timeout":
        return "Grok app-chat navigation timed out via Browser Bridge"
    if bridge_code == "sso_unavailable":
        return "Grok app-chat authentication failed via Browser Bridge"
    if bridge_code == "page_not_prepared":
        return "Grok app-chat Browser Bridge page was not prepared"
    if bridge_code in {"bridge_unavailable", "browser_bridge_unavailable"}:
        return "Grok app-chat Browser Bridge unavailable"
    return "Grok app-chat via Browser Bridge failed"


def _bridge_error_from_payload(
    payload: dict[str, Any] | None,
    upstream_status: int | None = None,
    default_code: str = "browser_bridge_unavailable",
) -> GrokConsoleError:
    data = payload if isinstance(payload, dict) else {}
    raw_code = data.get("code") or data.get("last_error_code") or default_code
    bridge_code = str(raw_code or default_code)
    status = _BRIDGE_ERROR_STATUS.get(bridge_code, _openai_status(upstream_status or 503))
    return GrokConsoleError(
        _safe_bridge_message(bridge_code),
        status,
        upstream_status,
        code=_browser_bridge_error_code(bridge_code),
        extra_detail={"bridge_code": bridge_code},
    )


def _preflight_browser_bridge_health(bridge_url: str) -> None:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{bridge_url}/health", timeout=_BRIDGE_HEALTH_TIMEOUT) as response:
            health = _parse_bridge_json_response(response)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise GrokConsoleError(
            "Grok app-chat Browser Bridge unavailable",
            503,
            None,
            code="grok_browser_bridge_unavailable",
            extra_detail={"bridge_code": "browser_bridge_unavailable"},
        ) from exc
    if health.get("status") != "ok":
        raise _bridge_error_from_payload(health, 503, "browser_bridge_unavailable")


def app_chat_headers(access_token: str, account: dict[str, Any] | None = None) -> dict[str, str]:
    profile = _grok_app_chat_profile()
    headers = {
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7",
        "Content-Type": "application/json",
        "Origin": APP_CHAT_BASE_URL,
        "Referer": f"{APP_CHAT_BASE_URL}/",
        "Priority": "u=1, i",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": _app_chat_profile_value(profile, account, "user_agent", "user_agent", "user-agent"),
        "x-statsig-id": _app_chat_profile_value(profile, account, "statsig_id", "statsig_id", "x-statsig-id"),
        "x-xai-request-id": str(uuid.uuid4()),
    }
    sec_ch_ua = _app_chat_profile_value(profile, account, "sec_ch_ua", "sec_ch_ua", "sec-ch-ua")
    sec_ch_ua_mobile = _app_chat_profile_value(profile, account, "sec_ch_ua_mobile", "sec_ch_ua_mobile", "sec-ch-ua-mobile")
    sec_ch_ua_platform = _app_chat_profile_value(profile, account, "sec_ch_ua_platform", "sec_ch_ua_platform", "sec-ch-ua-platform")
    if sec_ch_ua:
        headers["Sec-Ch-Ua"] = sec_ch_ua
    if sec_ch_ua_mobile:
        headers["Sec-Ch-Ua-Mobile"] = sec_ch_ua_mobile
    if sec_ch_ua_platform:
        headers["Sec-Ch-Ua-Platform"] = sec_ch_ua_platform
    cf_clearance = _app_chat_profile_value(profile, account, "cf_clearance", "cf_clearance")
    cf_cookies = _app_chat_profile_value(profile, account, "cf_cookies", "cf_cookies")
    cookie = _app_chat_cookie(access_token, cf_clearance, cf_cookies)
    if cookie:
        headers["Cookie"] = cookie
    return headers


def latest_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if str(message.get("role") or "user").strip().lower() != "user":
            continue
        text = _message_text(message.get("content"), "user").strip()
        if text:
            return text
    return ""


def grok_image_aspect_ratio(size: object) -> str:
    return GROK_IMAGE_ASPECT_RATIOS.get(str(size or "").strip().lower(), "")


def grok_image_response_format(value: object, default: str = "b64_json") -> str:
    if value is None or value == "":
        response_format = default
    else:
        response_format = str(value).strip().lower()
    if response_format in GROK_IMAGE_RESPONSE_FORMATS:
        return response_format
    raise ImageGenerationError(
        "Grok image response_format must be one of base64, b64_json, url",
        status_code=400,
        error_type="invalid_request_error",
        code="invalid_response_format",
        param="response_format",
    )


def build_app_chat_payload(
    spec: ModelSpec,
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    image_generation: bool = False,
) -> dict[str, Any]:
    message = str(body.get("prompt") or "").strip() or latest_user_message(messages)
    if not message:
        raise HTTPException(status_code=400, detail={"error": "Grok app-chat requires a user text message"})
    payload: dict[str, Any] = {
        "collectionIds": [],
        "connectors": [],
        "deviceEnvInfo": {
            "darkModeEnabled": False,
            "devicePixelRatio": 2,
            "screenHeight": 1329,
            "screenWidth": 2056,
            "viewportHeight": 1083,
            "viewportWidth": 2056,
        },
        "disableMemory": True,
        "disableSearch": False,
        "disableSelfHarmShortCircuit": False,
        "disableTextFollowUps": False,
        "enableImageGeneration": image_generation,
        "enableImageStreaming": image_generation,
        "enableSideBySide": True,
        "fileAttachments": [],
        "forceConcise": False,
        "forceSideBySide": False,
        "imageAttachments": [],
        "imageGenerationCount": int(body.get("n") or 1) if image_generation else 0,
        "isAsyncChat": False,
        "message": message,
        "modeId": spec.mode_id or spec.upstream_model or spec.id,
        "responseMetadata": {},
        "returnImageBytes": False,
        "returnRawGrokInXaiRequest": False,
        "searchAllConnectors": False,
        "sendFinalMetadata": True,
        "temporary": True,
        "toolOverrides": {
            "imageGen": False,
            "webSearch": not image_generation,
            "xSearch": False,
            "xMediaSearch": False,
            "trendsSearch": False,
            "xPostAnalyze": False,
        },
    }
    if spec.model_tier:
        payload["modelTier"] = spec.model_tier
    if spec.prefer_best:
        payload["preferBest"] = True
    if image_generation:
        aspect_ratio = grok_image_aspect_ratio(body.get("size"))
        if aspect_ratio:
            payload["modelConfigOverride"] = {
                "modelMap": {
                    "imageGenModel": spec.upstream_model or spec.id,
                    "imageGenModelConfig": {
                        "aspectRatio": aspect_ratio,
                    },
                }
            }
    return payload


def parse_app_chat_payload_line(line: str | bytes) -> dict[str, Any] | None:
    if isinstance(line, bytes):
        payload = line.decode("utf-8", errors="replace").strip()
    else:
        payload = str(line or "").strip()
    if not payload or payload == "[DONE]":
        return None
    if payload.startswith(("event:", "id:", "retry:")):
        return None
    if payload.startswith("data:"):
        payload = payload[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    nested = data.get("data")
    if isinstance(nested, dict) and not any(key in data for key in ("result", "response", "token", "finalMetadata", "usage")):
        return nested
    return data


def _decode_app_chat_line(line: str | bytes) -> str:
    return line.decode("utf-8", errors="replace") if isinstance(line, bytes) else str(line or "")


def _split_app_chat_embedded_data(raw: str) -> list[str]:
    if "data:" not in raw:
        return [raw]
    parts: list[str] = []
    current: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                parts.append("\n".join(current))
                parts.append("")
                current = []
            continue
        if stripped.startswith("data:"):
            if stripped.count("data:") > 1:
                for frame in re.split(r"(?=data:)", stripped):
                    if not frame:
                        continue
                    if current:
                        parts.append("\n".join(current))
                        parts.append("")
                    current = [frame]
                if current:
                    parts.append("\n".join(current))
                    current = []
                continue
            if current:
                parts.append("\n".join(current))
            current = [line]
            continue
        if "data:" in line:
            prefix, suffix = line.split("data:", 1)
            if prefix.strip():
                current.append(prefix)
            if current:
                parts.append("\n".join(current))
                parts.append("")
            current = ["data:" + suffix]
            continue
        current.append(line)
    if current:
        parts.append("\n".join(current))
    return parts


def _parse_app_chat_data_parts(data_parts: list[str]) -> dict[str, Any] | None:
    if not data_parts:
        return None
    event = parse_app_chat_payload_line("data: " + "\n".join(data_parts))
    if event is not None:
        return event
    if len(data_parts) > 1:
        return parse_app_chat_payload_line("data: " + "".join(data_parts))
    return None


def app_chat_line_events(lines: Iterable[str | bytes]) -> Iterator[dict[str, Any]]:
    data_parts: list[str] = []
    for line in lines:
        raw = _decode_app_chat_line(line)
        for part in _split_app_chat_embedded_data(raw):
            stripped = part.strip()
            if not stripped:
                event = _parse_app_chat_data_parts(data_parts)
                data_parts.clear()
                if event is not None:
                    yield event
                continue
            if stripped.startswith("data:"):
                value = stripped[5:].strip()
                if value == "[DONE]":
                    event = _parse_app_chat_data_parts(data_parts)
                    data_parts.clear()
                    if event is not None:
                        yield event
                    continue
                parsed = parse_app_chat_payload_line(stripped)
                if parsed is not None:
                    event = _parse_app_chat_data_parts(data_parts)
                    data_parts.clear()
                    if event is not None:
                        yield event
                    yield parsed
                    continue
                data_parts.append(value)
                continue
            event = _parse_app_chat_data_parts(data_parts)
            data_parts.clear()
            if event is not None:
                yield event
            event = parse_app_chat_payload_line(stripped)
            if event is not None:
                yield event
    event = _parse_app_chat_data_parts(data_parts)
    if event is not None:
        yield event


def _event_result(event: dict[str, Any]) -> dict[str, Any]:
    result = event.get("result")
    if isinstance(result, dict):
        return result
    data = event.get("data")
    if isinstance(data, dict) and isinstance(data.get("result"), dict):
        return data["result"]
    return {}


def _event_response(event: dict[str, Any]) -> dict[str, Any]:
    result = _event_result(event)
    response = result.get("response")
    if isinstance(response, dict):
        return response
    response = event.get("response")
    if isinstance(response, dict):
        return response
    if any(key in event for key in ("token", "isThinking", "messageTag", "isSoftStop", "finalMetadata")):
        return event
    return {}


def _app_chat_metadata(event: dict[str, Any]) -> dict[str, Any]:
    result = _event_result(event)
    response = _event_response(event)
    for source in (response, result, event):
        for key in ("finalMetadata", "final_metadata", "responseMetadata", "metadata"):
            if isinstance(source, dict) and key in source:
                value = source.get(key)
                return value if isinstance(value, dict) else {}
    return {}


def _app_chat_has_final_metadata(event: dict[str, Any]) -> bool:
    result = _event_result(event)
    response = _event_response(event)
    return any(isinstance(source, dict) and any(key in source for key in ("finalMetadata", "final_metadata")) for source in (response, result, event))


def extract_app_chat_token(event: dict[str, Any]) -> tuple[str, bool]:
    response = _event_response(event)
    token = response.get("token") or response.get("text") or response.get("content") or event.get("token")
    if not token:
        return "", False
    message_tag = str(response.get("messageTag") or event.get("messageTag") or "").lower()
    thinking = response.get("isThinking") is True or event.get("isThinking") is True or message_tag in {"thinking", "reasoning"}
    return str(token), thinking


def is_app_chat_final_event(event: dict[str, Any]) -> bool:
    result = _event_result(event)
    response = _event_response(event)
    if response.get("isSoftStop") is True or result.get("isSoftStop") is True or event.get("isSoftStop") is True:
        return True
    if _app_chat_has_final_metadata(event):
        return True
    return False


def _app_chat_json_data(attachment: dict[str, Any]) -> dict[str, Any]:
    raw_json_data = attachment.get("jsonData")
    if isinstance(raw_json_data, dict):
        return raw_json_data
    if isinstance(raw_json_data, str):
        try:
            decoded = json.loads(raw_json_data)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _app_chat_attachment_json_values(attachment: dict[str, Any]) -> list[dict[str, Any]]:
    values = [_app_chat_json_data(attachment)]
    for key in ("json", "data", "metadata"):
        value = attachment.get(key)
        if isinstance(value, dict):
            values.append(value)
    return [value for value in values if value]


def _app_chat_image_chunk_values(event: dict[str, Any]) -> list[dict[str, Any]]:
    response = _event_response(event)
    values: list[dict[str, Any]] = []
    for key in ("streamingImageGenerationResponse", "imageGenerationResponse", "imageGeneration", "imageResponse", "image_chunk", "imageChunk"):
        value = response.get(key)
        if isinstance(value, dict):
            values.append(value)
    attachment_obj = response.get("cardAttachment")
    attachment: dict[str, Any] = attachment_obj if isinstance(attachment_obj, dict) else {}
    for json_data in _app_chat_attachment_json_values(attachment):
        for key in ("image_chunk", "imageChunk", "image", "media"):
            value = json_data.get(key)
            if isinstance(value, dict):
                values.append(value)
        values.append(json_data)
    metadata = _app_chat_metadata(event)
    for key in ("image_chunk", "imageChunk", "streamingImageGenerationResponse", "imageGenerationResponse", "imageGeneration", "imageResponse"):
        value = metadata.get(key)
        if isinstance(value, dict):
            values.append(value)
    for key in ("image", "media", "attachment"):
        value = event.get(key)
        if isinstance(value, dict):
            values.append(value)
    return values


def _is_moderated_or_blocked(value: dict[str, Any]) -> bool:
    return any(value.get(key) is True for key in ("moderated", "isModerated", "blocked", "isBlocked", "contentFiltered"))


def app_chat_moderation_message(event: dict[str, Any]) -> str:
    response = _event_response(event)
    candidates = [response, _event_result(event), event, _app_chat_metadata(event)]
    candidates.extend(_app_chat_image_chunk_values(event))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if not _is_moderated_or_blocked(candidate) and str(candidate.get("status") or "").lower() not in {"blocked", "moderated"}:
            continue
        for key in ("message", "error", "reason", "blockReason", "moderationReason", "statusMessage"):
            value = candidate.get(key)
            if value:
                return str(value)
        return "Grok blocked this image request"
    return ""


def _resolve_app_chat_image_url(value: object) -> str:
    resolved = _resolve_grok_asset_url(str(value or ""))
    if resolved:
        return resolved
    text = str(value or "").strip()
    if text and not urlparse(text).scheme:
        return GROK_ASSET_BASE_URL + text.lstrip("/")
    return ""


def extract_app_chat_image_urls(event: dict[str, Any]) -> list[str]:
    image_urls: list[str] = []
    seen: set[str] = set()
    for chunk in _app_chat_image_chunk_values(event):
        try:
            progress = int(chunk.get("progress") or 100)
        except (TypeError, ValueError):
            progress = 0
        if progress < 100 or _is_moderated_or_blocked(chunk):
            continue
        for key in ("imageUrl", "imageURL", "url", "mediaUrl", "mediaURL", "generatedImageUrl", "generatedImageURL", "assetUrl", "assetURL", "finalUrl", "finalURL", "contentUrl", "contentURL"):
            image_url = _resolve_app_chat_image_url(chunk.get(key))
            if image_url and image_url not in seen:
                image_urls.append(image_url)
                seen.add(image_url)
        image_url = resolve_grok_asset_reference(str(chunk.get("assetId") or chunk.get("asset_id") or chunk.get("fileId") or chunk.get("file_id") or "").strip(), "", str(chunk.get("userId") or chunk.get("user_id") or ""))
        if image_url and image_url not in seen:
            image_urls.append(image_url)
            seen.add(image_url)
    return image_urls


def extract_app_chat_image_url(event: dict[str, Any]) -> str:
    urls = extract_app_chat_image_urls(event)
    return urls[0] if urls else ""


def _search_source_text(value: object) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return " ".join(" ".join(parts).split())
    return ""


def _search_source_title(value: object, fallback: str) -> str:
    title = _search_source_text(value)
    if not title:
        return fallback
    return title[:80].rstrip()


def _app_chat_search_container(event: dict[str, Any], key: str) -> dict[str, Any]:
    result = _event_result(event)
    response = _event_response(event)
    metadata = _app_chat_metadata(event)
    for source in (response, metadata, result, event):
        if isinstance(source, dict):
            value = source.get(key)
            if isinstance(value, dict):
                return value
            if isinstance(value, list):
                return {"results": value}
    return {}


def _app_chat_search_results(event: dict[str, Any], key: str, aliases: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name in (key, *aliases):
        raw = _app_chat_search_container(event, name).get("results")
        if isinstance(raw, list):
            results.extend(item for item in raw if isinstance(item, dict))
    return results


def extract_app_chat_search_sources(event: dict[str, Any]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    for item in _app_chat_search_results(event, "webSearchResults", ("webResults", "sources", "citations")):
        url = str(item.get("url") or item.get("link") or item.get("sourceUrl") or "").strip()
        if not url:
            continue
        sources.append({
            "url": url,
            "title": _search_source_title(item.get("title") or item.get("name") or item.get("text"), url),
            "type": "web",
        })
    for item in _app_chat_search_results(event, "xSearchResults", ("xResults", "xPostResults")):
        username = str(item.get("username") or item.get("screenName") or item.get("userName") or "").strip().lstrip("@")
        post_id = str(item.get("postId") or item.get("id") or item.get("restId") or "").strip()
        url = str(item.get("url") or item.get("link") or "").strip()
        if not url and username and post_id:
            url = f"https://x.com/{username}/status/{post_id}"
        if not url:
            continue
        sources.append({
            "url": url,
            "title": _search_source_title(item.get("text") or item.get("fullText") or item.get("content") or item.get("title"), f"@{username}" if username else url),
            "type": "x_post",
        })
    return sources


def dedupe_search_sources(sources: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in sources:
        url = str(source.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append({
            "url": url,
            "title": str(source.get("title") or url).strip() or url,
            "type": str(source.get("type") or "web").strip() or "web",
        })
    return deduped


def _markdown_link_text(value: object) -> str:
    return str(value or "Source").replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]").strip() or "Source"


def _markdown_link_url(value: object) -> str:
    url = str(value or "").strip()
    if not url or any(ord(char) < 32 or char.isspace() or char == ")" for char in url):
        return ""
    return url


def format_search_sources_suffix(sources: Iterable[dict[str, str]]) -> str:
    deduped = dedupe_search_sources(sources)
    if not deduped:
        return ""
    lines = ["", "", SEARCH_SOURCES_MARKER, "## Sources"]
    for source in deduped:
        title = _markdown_link_text(source.get("title") or source.get("url") or "Source")
        url = _markdown_link_url(source.get("url"))
        if url:
            lines.append(f"{len(lines) - 3}. [{title}]({url})")
    return "\n".join(lines) if len(lines) > 4 else ""


def append_search_sources_suffix(content: str, sources: Iterable[dict[str, str]]) -> str:
    suffix = format_search_sources_suffix(sources)
    return f"{content}{suffix}" if suffix else content


def strip_search_sources_suffix(content: str) -> str:
    return _SEARCH_SOURCES_BLOCK_RE.sub("", content).rstrip()


def strip_search_sources_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            stripped.append(message)
            continue
        next_message = dict(message)
        content = next_message.get("content")
        if isinstance(content, str):
            next_message["content"] = strip_search_sources_suffix(content)
        elif isinstance(content, list):
            blocks: list[Any] = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    next_block = dict(block)
                    next_block["text"] = strip_search_sources_suffix(next_block["text"])
                    blocks.append(next_block)
                else:
                    blocks.append(block)
            next_message["content"] = blocks
        stripped.append(next_message)
    return stripped


def collect_app_chat_response(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    search_sources: list[dict[str, str]] = []
    for event in events:
        search_sources.extend(extract_app_chat_search_sources(event))
        token, thinking = extract_app_chat_token(event)
        if token:
            if thinking:
                reasoning_parts.append(token)
            else:
                content_parts.append(token)
        if is_app_chat_final_event(event):
            break
    return {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "search_sources": dedupe_search_sources(search_sources),
    }


def _safe_app_chat_error_text(response: object | None = None, detail: object | None = None) -> str:
    parts: list[str] = []
    if detail is not None:
        parts.append(str(detail))
    if response is not None:
        json_method = getattr(response, "json", None)
        if callable(json_method):
            try:
                data = json_method()
            except Exception:
                data = None
            if isinstance(data, dict):
                parts.append(json.dumps(data, ensure_ascii=False))
        text = getattr(response, "text", "") or ""
        if not text:
            content = getattr(response, "content", b"")
            if isinstance(content, bytes):
                text = content.decode("utf-8", errors="replace")
        if text:
            parts.append(str(text))
    text = " ".join(parts).strip().lower()
    return _redact_grok_secrets(text)[:2000]


def _redact_grok_secrets(text: str) -> str:
    redacted = re.sub(r"(?i)(sso(?:-rw)?\s*=\s*)[^;\s<>'\"]+", r"\1[redacted]", text)
    redacted = re.sub(r"(?i)((?:cf_clearance|__cf_bm|cf_bm)\s*=\s*)[^;\s<>'\"]+", r"\1[redacted]", redacted)
    redacted = re.sub(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^;\s<>'\"]+", r"\1[redacted]", redacted)
    return redacted


def _safe_exception_message(exc: BaseException) -> str:
    return _redact_grok_secrets(str(exc))[:1000]


def _app_chat_error_contains(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _app_chat_error_field_text(value: object) -> str:
    parts: list[str] = []
    if isinstance(value, dict):
        for key in ("error", "code", "type", "reason", "message", "detail", "status"):
            item = value.get(key)
            if isinstance(item, (str, int, float)):
                parts.append(str(item))
            elif isinstance(item, dict):
                parts.append(_app_chat_error_field_text(item))
        for item in value.values():
            if isinstance(item, (dict, list)):
                parts.append(_app_chat_error_field_text(item))
    elif isinstance(value, list):
        for item in value:
            parts.append(_app_chat_error_field_text(item))
    elif isinstance(value, (str, int, float)):
        parts.append(str(value))
    return " ".join(part for part in parts if part).lower()


def _app_chat_structured_error_text(response: object | None = None, detail: object | None = None) -> str:
    values: list[object] = []
    if detail is not None:
        values.append(detail)
    if response is not None:
        json_method = getattr(response, "json", None)
        if callable(json_method):
            try:
                values.append(json_method())
            except Exception:
                pass
    return " ".join(_app_chat_error_field_text(value) for value in values).strip()


def _is_app_chat_challenge_error(status: int, text: str, structured_text: str = "") -> bool:
    return status in _APP_CHAT_AUTH_STATUS_CODES and _app_chat_error_contains(text, _APP_CHAT_CHALLENGE_MARKERS) and not _app_chat_error_contains(structured_text, _APP_CHAT_AUTH_ERROR_MARKERS)


def _is_app_chat_auth_error(status: int, text: str, structured_text: str = "") -> bool:
    auth_text = structured_text or text
    return status in _APP_CHAT_AUTH_STATUS_CODES and _app_chat_error_contains(auth_text, _APP_CHAT_AUTH_ERROR_MARKERS)


def _is_app_chat_limit_error(status: int, text: str, structured_text: str = "") -> bool:
    return status in _APP_CHAT_LIMIT_STATUS_CODES or _app_chat_error_contains(structured_text or text, _APP_CHAT_RATE_LIMIT_MARKERS)


def _is_app_chat_transient_error(status: int, text: str, structured_text: str = "") -> bool:
    return status in _APP_CHAT_TRANSIENT_STATUS_CODES or _app_chat_error_contains(structured_text or text, _APP_CHAT_TRANSIENT_ERROR_MARKERS)


def classify_app_chat_upstream_error(upstream_status: int, access_token: str | None = None, response: object | None = None, detail: object | None = None) -> GrokConsoleError:
    status = int(upstream_status)
    text = _safe_app_chat_error_text(response, detail)
    structured_text = _app_chat_structured_error_text(response, detail)
    is_limit = _is_app_chat_limit_error(status, text, structured_text)
    is_challenge = _is_app_chat_challenge_error(status, text, structured_text)
    is_auth = _is_app_chat_auth_error(status, text, structured_text)
    is_transient = _is_app_chat_transient_error(status, text, structured_text)
    feedback_status: str | None = None
    if is_limit:
        feedback_status = "限流"
    elif is_auth and not is_challenge:
        feedback_status = "异常"
    if access_token and feedback_status:
        from services.account_service import account_service

        account_service.update_account(access_token, {"status": feedback_status})
    if is_limit:
        return GrokConsoleError(f"Grok app-chat rate limited (HTTP {status})", 429, status, "rate_limit_exceeded")
    if is_challenge:
        return GrokConsoleError(f"Grok app-chat Cloudflare challenge blocked request (HTTP {status})", 502, status, "cloudflare_challenge")
    if is_auth:
        return GrokConsoleError(f"Grok app-chat authentication failed (HTTP {status})", 401, status, "authentication_failed")
    if is_transient:
        return GrokConsoleError(f"Grok app-chat transient upstream error (HTTP {status})", 502, status, "upstream_transient")
    if status == 403:
        return GrokConsoleError("Grok app-chat forbidden (HTTP 403)", 502, status, "rate_limit_check_unavailable")
    return GrokConsoleError(f"{_APP_CHAT_PUBLIC_ERROR_PREFIX} (HTTP {status})", _openai_status(status), status)


def _is_safe_grok_asset_path(value: str) -> bool:
    if not value:
        return False
    if value.startswith("//") or "\\" in value:
        return False
    if any(ord(char) < 32 for char in value):
        return False
    if ":" in value.split("/", 1)[0]:
        return False
    return True


def _resolve_grok_asset_url(value: str) -> str:
    url = str(value or "").strip()
    if not url:
        return ""
    if url.startswith(("http://", "https://")):
        parsed = urlparse(url)
        if parsed.scheme == "https" and parsed.netloc == "assets.grok.com":
            return url
        return ""
    if not _is_safe_grok_asset_path(url):
        return ""
    return GROK_ASSET_BASE_URL + url.lstrip("/")


def _extract_x_user_id(access_token: str) -> str:
    for name, value in _cookie_items(_app_chat_cookie(access_token)):
        if name == "x-userid":
            return value
    return ""


def resolve_grok_asset_reference(file_id: str, file_uri: str = "", user_id: str = "") -> str:
    if file_uri:
        return _resolve_grok_asset_url(file_uri)
    if file_id and user_id:
        return f"{GROK_ASSET_BASE_URL}users/{user_id}/{file_id}/content"
    return ""


def _invalid_image_edit_request(message: str, param: str | None = None) -> ImageGenerationError:
    return ImageGenerationError(
        message,
        status_code=400,
        error_type="invalid_request_error",
        code="invalid_request_error",
        param=param,
    )


def validate_grok_image_edit_request(
    images: list[tuple[bytes, str, str]],
    n: int,
    size: str | None,
) -> None:
    normalized_size = str(size or GROK_IMAGE_EDIT_SIZE).strip().lower()
    if normalized_size != GROK_IMAGE_EDIT_SIZE:
        raise _invalid_image_edit_request(
            f"Grok image edit only supports size {GROK_IMAGE_EDIT_SIZE}",
            "size",
        )
    if n < 1 or n > GROK_IMAGE_EDIT_MAX_N:
        raise _invalid_image_edit_request("Grok image edit supports n between 1 and 2", "n")
    if not images:
        raise _invalid_image_edit_request("image is required", "image")
    if len(images) > GROK_IMAGE_EDIT_MAX_REFERENCES:
        raise _invalid_image_edit_request("Grok image edit supports at most 7 reference images", "image")


def build_grok_image_edit_payload(prompt: str, image_references: list[str], parent_post_id: str) -> dict[str, Any]:
    return {
        "temporary": True,
        "modelName": GROK_IMAGE_EDIT_MODEL_NAME,
        "message": prompt,
        "enableImageGeneration": True,
        "returnImageBytes": False,
        "returnRawGrokInXaiRequest": False,
        "enableImageStreaming": True,
        "imageGenerationCount": GROK_IMAGE_EDIT_MAX_N,
        "forceConcise": False,
        "enableSideBySide": True,
        "sendFinalMetadata": True,
        "isReasoning": False,
        "disableTextFollowUps": True,
        "responseMetadata": {
            "modelConfigOverride": {
                "modelMap": {
                    "imageEditModel": GROK_IMAGE_EDIT_MODEL_KIND,
                    "imageEditModelConfig": {
                        "imageReferences": image_references,
                        "parentPostId": parent_post_id,
                    },
                }
            }
        },
        "disableMemory": True,
        "forceSideBySide": False,
    }


def build_grok_media_post_payload(media_type: str, media_url: str = "", prompt: str = "") -> dict[str, Any]:
    payload: dict[str, Any] = {"mediaType": media_type}
    if media_url:
        payload["mediaUrl"] = media_url
    if prompt:
        payload["prompt"] = prompt
    return payload


def replace_grok_image_placeholders(prompt: str, references: list[GrokImageEditReference]) -> str:
    def replace(match: re.Match[str]) -> str:
        image_number = int(match.group(1))
        if image_number < 1 or image_number > len(references):
            return match.group(0)
        return f"@{references[image_number - 1].file_id}"

    return _GROK_IMAGE_PLACEHOLDER_RE.sub(replace, prompt)


def _extract_grok_streaming_image_response(event: dict[str, Any]) -> dict[str, Any]:
    result_obj = event.get("result")
    result = result_obj if isinstance(result_obj, dict) else {}
    response_obj = result.get("response")
    response = response_obj if isinstance(response_obj, dict) else {}
    stream = response.get("streamingImageGenerationResponse")
    return stream if isinstance(stream, dict) else {}


def _extract_grok_model_response(event: dict[str, Any]) -> dict[str, Any]:
    result_obj = event.get("result")
    result = result_obj if isinstance(result_obj, dict) else {}
    response_obj = result.get("response")
    response = response_obj if isinstance(response_obj, dict) else {}
    model_response = response.get("modelResponse")
    return model_response if isinstance(model_response, dict) else {}


def extract_grok_image_edit_final_urls(event: dict[str, Any], user_id: str = "") -> dict[int, str]:
    urls: dict[int, str] = {}
    for stream in _app_chat_image_chunk_values(event):
        try:
            progress = int(stream.get("progress") or 0)
        except (TypeError, ValueError):
            progress = 0
        if progress >= 100 and not _is_moderated_or_blocked(stream):
            resolved = ""
            for key in ("imageUrl", "imageURL", "url", "mediaUrl", "mediaURL", "generatedImageUrl", "generatedImageURL", "assetUrl", "assetURL", "finalUrl", "finalURL", "contentUrl", "contentURL"):
                resolved = _resolve_app_chat_image_url(stream.get(key))
                if resolved:
                    break
            if not resolved:
                resolved = resolve_grok_asset_reference(
                    str(stream.get("assetId") or stream.get("asset_id") or stream.get("fileId") or stream.get("file_id") or "").strip(),
                    "",
                    user_id or str(stream.get("userId") or stream.get("user_id") or ""),
                )
            if resolved:
                try:
                    index = int(stream.get("imageIndex") or 0)
                except (TypeError, ValueError):
                    index = 0
                if index >= 0:
                    urls[index] = resolved
    model_response = _extract_grok_model_response(event)
    attachments = model_response.get("fileAttachments")
    if isinstance(attachments, list):
        for index, asset_id in enumerate(attachments):
            resolved = resolve_grok_asset_reference(str(asset_id or "").strip(), "", user_id)
            if resolved:
                urls.setdefault(index, resolved)
    generated_urls = model_response.get("generatedImageUrls") or model_response.get("generatedImages")
    if isinstance(generated_urls, list):
        for index, item in enumerate(generated_urls):
            if isinstance(item, dict):
                raw_url = item.get("url") or item.get("imageUrl") or item.get("assetUrl") or item.get("contentUrl")
            else:
                raw_url = item
            resolved = _resolve_grok_asset_url(str(raw_url or ""))
            if resolved:
                urls.setdefault(index, resolved)
    return urls


class GrokAppChatClient:
    def __init__(self, access_token: str, account: dict[str, Any] | None = None) -> None:
        self.access_token = access_token
        self.account = account if isinstance(account, dict) else None
        self.network_profile = _grok_app_chat_profile()
        impersonate = _app_chat_impersonate(self.network_profile, self.account)
        self.session = create_session(impersonate=impersonate, verify=self.network_profile.verify, account=self.account)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "GrokAppChatClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _call_with_retry(self, fn, policy=None, context=""):
        from services.network.retry import retry_call, RetryPolicy

        api_policy = policy or RetryPolicy(
            max_attempts=3,
            retry_statuses=frozenset({408, 429, 500, 502, 503, 504}),
        )
        deadline = time.monotonic() + 60.0

        def on_retry(attempt, status_code, exc):
            error = _safe_exception_message(exc) if exc else None
            logger.warning({
                "event": "grok_app_chat_retry",
                "context": context,
                "attempt": attempt,
                "status_code": status_code,
                "error": error,
            })

        return retry_call(fn, policy=api_policy, deadline=deadline, on_retry=on_retry)

    def validate_rate_limits(self) -> dict[str, Any]:
        try:
            response = self._call_with_retry(
                lambda: self.session.post(
                    APP_CHAT_RATE_LIMITS_URL,
                    headers=app_chat_headers(self.access_token, self.account),
                    json=build_grok_rate_limits_payload(self.account),
                    timeout=self.network_profile.timeout,
                ),
                context="app_chat_rate_limits",
            )
        except requests.exceptions.RequestException as exc:
            raise GrokConsoleError(
                "Grok app-chat rate-limit check unavailable",
                502,
                code="rate_limit_network_error",
            ) from exc
        if response.status_code >= 400:
            error = classify_app_chat_upstream_error(int(response.status_code), self.access_token, response)
            if error.upstream_status in {401, 403} and error.code is None:
                raise GrokConsoleError(
                    f"Grok app-chat rate-limit check unavailable (HTTP {error.upstream_status})",
                    502,
                    error.upstream_status,
                    "rate_limit_check_unavailable",
                ) from error
            raise error
        try:
            data = response.json()
        except Exception as exc:
            raise GrokConsoleError(
                "Grok app-chat rate-limit validation returned an invalid response",
                502,
                code="invalid_rate_limit_response",
            ) from exc
        if not isinstance(data, dict):
            raise GrokConsoleError(
                "Grok app-chat rate-limit validation returned an invalid response",
                502,
                code="invalid_rate_limit_response",
            )
        try:
            from services.account_service import account_service
            from services.providers.grok.accounts import normalize_app_chat_rate_limit_payload

            updates = grok_rate_limit_account_hints(data)
            quota_payload = normalize_app_chat_rate_limit_payload(data)
            if quota_payload:
                updates["quota_console"] = quota_payload
            if updates:
                account_service.update_account(self.access_token, updates)
        except Exception:
            logger.debug({"event": "grok_app_chat_rate_limit_hints_skipped"})
        return data

    def _refresh_clearance(self) -> bool:
        if not config.flaresolverr_url:
            return False
        with _APP_CHAT_CLEARANCE_LOCK:
            clearance = FlareSolverrClearanceProvider().solve()
            if clearance is None:
                return False
            current_profiles = cast(dict[str, Any], config.network_profiles)
            app_profile = dict(cast(dict[str, Any], current_profiles.get("grok_app_chat") or {}))
            solved_impersonate = infer_chromium_impersonate(clearance.user_agent)
            app_profile.update({
                "browser": solved_impersonate or app_profile.get("browser") or app_profile.get("impersonate"),
                "cf_cookies": clearance.cf_cookies,
                "cf_clearance": clearance.cf_clearance,
                "impersonate": solved_impersonate or app_profile.get("impersonate") or app_profile.get("browser"),
                "user-agent": clearance.user_agent,
            })
            next_profiles = dict(current_profiles)
            next_profiles["grok_app_chat"] = app_profile
            config.update({"network_profiles": next_profiles})
            self.network_profile = _grok_app_chat_profile()
            headers = app_chat_headers(self.access_token, self.account)
            session_headers = getattr(self.session, "headers", None)
            if session_headers is not None and hasattr(session_headers, "update"):
                session_headers.update(headers)
            return True

    def _try_browser_bridge(self, payload: dict[str, Any]) -> list[str] | None:
        configured_bridge = bool(config.browser_bridge_url)
        try:
            bridge_url = _detect_bridge_url()
        except GrokConsoleError:
            if configured_bridge:
                raise
            return None
        if not bridge_url:
            return None
        sso = _extract_raw_sso(self.access_token)
        if not sso:
            return None
        import urllib.request
        import urllib.error
        if configured_bridge:
            _preflight_browser_bridge_health(bridge_url)
        bridge_body = json.dumps({"sso": sso, "payload": payload}).encode()
        req = urllib.request.Request(
            f"{bridge_url}/api/chat",
            data=bridge_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout = _BRIDGE_EXPLICIT_CHAT_TIMEOUT if configured_bridge else 120
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.warning({"event": "browser_bridge_error", "status": resp.status})
                    if configured_bridge:
                        raise _bridge_error_from_payload(_parse_bridge_json_response(resp), resp.status)
                    return None
                return resp.read().decode("utf-8", errors="replace").splitlines()
        except urllib.error.HTTPError as exc:
            payload = _parse_bridge_json_response(exc)
            if payload.get("code"):
                raise _bridge_error_from_payload(payload, exc.code) from exc
            msg = f"Grok app-chat forbidden (HTTP {exc.code}): account may lack required tier" if exc.code == 403 else f"Grok app-chat via bridge failed (HTTP {exc.code})"
            raise GrokConsoleError(msg, _openai_status(exc.code), exc.code) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            global _bridge_probed
            _bridge_probed = False
            logger.warning({"event": "browser_bridge_unavailable"})
            if configured_bridge:
                raise GrokConsoleError(
                    "Grok app-chat Browser Bridge unavailable",
                    503,
                    None,
                    code="grok_browser_bridge_unavailable",
                    extra_detail={"bridge_code": "browser_bridge_unavailable"},
                ) from exc
            return None

    def _stream_direct_events(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        try:
            response = self._call_with_retry(
                lambda: self.session.post(
                    APP_CHAT_NEW_CONVERSATION_URL,
                    headers=app_chat_headers(self.access_token, self.account),
                    json=payload,
                    timeout=self.network_profile.timeout,
                    stream=True,
                ),
                context="app_chat",
            )
        except requests.exceptions.RequestException as exc:
            raise GrokConsoleError(f"Grok app-chat upstream request failed: {_safe_exception_message(exc)}", 502) from exc
        if response.status_code == 403 and self._refresh_clearance():
            try:
                response = self._call_with_retry(
                    lambda: self.session.post(
                        APP_CHAT_NEW_CONVERSATION_URL,
                        headers=app_chat_headers(self.access_token, self.account),
                        json=payload,
                        timeout=self.network_profile.timeout,
                        stream=True,
                    ),
                    context="app_chat_flaresolverr",
                )
            except requests.exceptions.RequestException as exc:
                raise GrokConsoleError(f"Grok app-chat upstream request failed: {_safe_exception_message(exc)}", 502) from exc
        if response.status_code >= 400:
            raise classify_app_chat_upstream_error(int(response.status_code), self.access_token, response)
        try:
            yield from app_chat_line_events(response.iter_lines())
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    def stream_events(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        bridge_first = bool(config.browser_bridge_url)
        if bridge_first:
            bridge_lines = self._try_browser_bridge(payload)
            if bridge_lines is not None:
                yield from app_chat_line_events(bridge_lines)
                return
        try:
            yield from self._stream_direct_events(payload)
            return
        except GrokConsoleError as exc:
            status = exc.upstream_status or exc.status_code
            if bridge_first or status not in {408, 502, 503, 504}:
                raise
            bridge_lines = self._try_browser_bridge(payload)
            if bridge_lines is None:
                raise
            logger.info({"event": "grok_app_chat_direct_fallback_to_bridge", "status": status})
            yield from app_chat_line_events(bridge_lines)

    def _post_direct_json(self, url: str, payload: dict[str, Any], *, context: str, referer: str | None = None) -> dict[str, Any]:
        headers = app_chat_headers(self.access_token, self.account)
        if referer:
            headers["Referer"] = referer
        try:
            response = self._call_with_retry(
                lambda: self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.network_profile.timeout,
                ),
                context=context,
            )
        except requests.exceptions.RequestException as exc:
            raise GrokConsoleError(f"Grok app-chat {context} request failed: {_safe_exception_message(exc)}", 502) from exc
        if response.status_code == 403 and self._refresh_clearance():
            headers = app_chat_headers(self.access_token, self.account)
            if referer:
                headers["Referer"] = referer
            try:
                response = self._call_with_retry(
                    lambda: self.session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.network_profile.timeout,
                    ),
                    context=f"{context}_flaresolverr",
                )
            except requests.exceptions.RequestException as exc:
                raise GrokConsoleError(f"Grok app-chat {context} request failed: {_safe_exception_message(exc)}", 502) from exc
        if response.status_code >= 400:
            raise classify_app_chat_upstream_error(int(response.status_code), self.access_token, response)
        try:
            data = response.json()
        except Exception as exc:
            raise GrokConsoleError(f"Grok app-chat {context} returned an invalid response", 502) from exc
        if not isinstance(data, dict):
            raise GrokConsoleError(f"Grok app-chat {context} returned an invalid response", 502)
        return data

    def upload_image_edit_reference(self, data: bytes, filename: str, mime_type: str) -> GrokImageEditReference:
        payload = {
            "fileName": filename or "image.png",
            "fileMimeType": mime_type or "application/octet-stream",
            "content": base64.b64encode(data).decode("ascii"),
        }
        result = self._post_direct_json(APP_CHAT_UPLOAD_FILE_URL, payload, context="upload_file")
        file_id = str(result.get("fileMetadataId") or result.get("fileMetadataID") or result.get("fileId") or result.get("fileID") or result.get("id") or "").strip()
        file_uri = str(result.get("fileUri") or result.get("fileURI") or result.get("contentUrl") or result.get("contentURL") or result.get("url") or "").strip()
        content_url = resolve_grok_asset_reference(file_id, file_uri, _extract_x_user_id(self.access_token))
        if not file_id or not content_url:
            raise GrokConsoleError("Grok image edit upload returned an invalid response", 502)
        return GrokImageEditReference(file_id=file_id, content_url=content_url)

    def create_image_edit_parent_post(self, prompt: str) -> tuple[str, str]:
        result = self._post_direct_json(
            APP_CHAT_MEDIA_POST_CREATE_URL,
            build_grok_media_post_payload(GROK_IMAGE_EDIT_MEDIA_TYPE, prompt=prompt),
            context="media_post_create",
            referer=f"{APP_CHAT_BASE_URL}/imagine",
        )
        post = result.get("post")
        if not isinstance(post, dict):
            raise GrokConsoleError("Grok image edit create-post returned no post payload", 502)
        post_id = str(post.get("id") or "").strip()
        if not post_id:
            raise GrokConsoleError("Grok image edit create-post returned no post id", 502)
        post_prompt = str(post.get("originalPrompt") or post.get("prompt") or "").strip()
        return post_id, post_prompt or prompt

    def stream_image_edit_events(
        self,
        prompt: str,
        image_references: list[str],
        parent_post_id: str,
    ) -> Iterator[dict[str, Any]]:
        payload = build_grok_image_edit_payload(prompt, image_references, parent_post_id)
        yield from self.stream_events(payload)


def validate_grok_access_token(access_token: str, account: dict[str, Any] | None = None) -> dict[str, Any]:
    with GrokAppChatClient(access_token, account) as client:
        return client.validate_rate_limits()


def _is_retryable_app_chat_account_error(exc: GrokConsoleError) -> bool:
    if exc.code in {"rate_limit_exceeded", "upstream_transient"}:
        return True
    status = exc.upstream_status or exc.status_code
    return int(status) in _APP_CHAT_RETRYABLE_STATUS_CODES


def _image_generation_error_from_grok(exc: GrokConsoleError) -> ImageGenerationError:
    return ImageGenerationError(str(exc), status_code=exc.status_code, code=exc.code)


def _no_available_grok_image_account_error(last_error: GrokConsoleError | None = None) -> ImageGenerationError:
    if last_error is not None:
        return _image_generation_error_from_grok(last_error)
    return ImageGenerationError("no available Grok account", status_code=503, code="no_available_account")


def app_chat_completion_events(body: dict[str, Any], spec: ModelSpec, messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    from services.account_service import account_service

    payload = build_app_chat_payload(spec, body, messages)
    excluded_tokens: set[str] = set()
    refreshed_after_empty = False
    last_error: GrokConsoleError | None = None

    for _ in range(_APP_CHAT_MAX_ACCOUNT_ATTEMPTS):
        access_token = account_service.get_grok_app_chat_access_token(spec, excluded_tokens=excluded_tokens)
        if not access_token:
            if not refreshed_after_empty:
                refreshed_after_empty = True
                try:
                    account_service.refresh_accounts([], provider=GROK_PROVIDER)
                except Exception as exc:
                    logger.warning({"event": "grok_app_chat_refresh_before_empty_failed", "error": _safe_exception_message(exc)})
                continue
            if last_error is not None:
                raise HTTPException(status_code=last_error.status_code, detail=last_error.to_http_detail()) from last_error
            raise HTTPException(status_code=503, detail={"error": "no available Grok account"})
        account = account_service.get_account(access_token)
        emitted_output = False
        try:
            with GrokAppChatClient(access_token, account) as client:
                for event in client.stream_events(payload):
                    emitted_output = True
                    yield event
        except GrokConsoleError as exc:
            if emitted_output or not _is_retryable_app_chat_account_error(exc):
                raise HTTPException(status_code=exc.status_code, detail=exc.to_http_detail()) from exc
            excluded_tokens.add(access_token)
            last_error = exc
            continue
        account_service.mark_text_used(access_token)
        return
    if last_error is not None:
        raise HTTPException(status_code=last_error.status_code, detail=last_error.to_http_detail()) from last_error
    raise HTTPException(status_code=503, detail={"error": "no available Grok account"})


def app_chat_completion(body: dict[str, Any], spec: ModelSpec, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return collect_app_chat_response(app_chat_completion_events(body, spec, messages))


def _grok_asset_base64(url: str) -> str:
    resolved_url = _resolve_grok_asset_url(url)
    if not resolved_url:
        raise ImageGenerationError(
            "Grok image base64 response_format only supports https://assets.grok.com assets",
            status_code=502,
            code="image_download_failed",
        )
    session = create_session()
    try:
        response = session.get(resolved_url, timeout=GROK_IMAGE_DOWNLOAD_TIMEOUT)
        if int(getattr(response, "status_code", 0) or 0) >= 400:
            raise ImageGenerationError(
                "Grok image asset download failed",
                status_code=502,
                code="image_download_failed",
            )
        content = getattr(response, "content", b"")
        if isinstance(content, str):
            content = content.encode("utf-8")
        elif not isinstance(content, bytes):
            content = bytes(content or b"")
        if not content:
            raise ImageGenerationError(
                "Grok image asset download returned an empty response",
                status_code=502,
                code="image_download_failed",
            )
        return base64.b64encode(content).decode("ascii")
    except ImageGenerationError:
        raise
    except requests.exceptions.RequestException as exc:
        raise ImageGenerationError(
            "Grok image asset download failed",
            status_code=502,
            code="image_download_failed",
        ) from exc
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()


def _format_grok_image_items(image_items: list[dict[str, Any]], response_format: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    for item in image_items:
        url = str(item.get("url") or "").strip()
        revised_prompt = str(item.get("revised_prompt") or "").strip()
        if response_format == "url":
            data.append({"url": url, "revised_prompt": revised_prompt})
            continue
        b64_json = _grok_asset_base64(url)
        if response_format == "base64":
            data.append({"base64": b64_json, "revised_prompt": revised_prompt})
        else:
            data.append({"b64_json": b64_json, "revised_prompt": revised_prompt})
    return data


def app_chat_image_outputs(body: dict[str, Any], spec: ModelSpec, prompt: str, n: int = 1) -> Iterator[ImageOutput]:
    if not is_supported_grok_app_chat_image_model(spec.id):
        if spec.capability == "video":
            detail = f"Grok video generation is not yet supported: {spec.id}"
        elif spec.capability == "image_edit":
            detail = f"Grok image editing is not yet supported: {spec.id}"
        else:
            detail = f"unsupported Grok image model: {spec.id}"
        raise ImageGenerationError(
            detail,
            status_code=400,
            error_type="invalid_request_error",
            code="unsupported_model",
            param="model",
        )
    request_body = dict(body)
    request_body["prompt"] = prompt
    request_body["n"] = n
    response_format = grok_image_response_format(request_body.get("response_format"))
    payload = build_app_chat_payload(spec, request_body, [{"role": "user", "content": prompt}], image_generation=True)

    from services.account_service import account_service

    excluded_tokens: set[str] = set()
    last_error: GrokConsoleError | None = None
    for _ in range(_APP_CHAT_MAX_ACCOUNT_ATTEMPTS):
        access_token = account_service.get_grok_app_chat_access_token(spec, excluded_tokens=excluded_tokens)
        if not access_token:
            raise _no_available_grok_image_account_error(last_error)
        account = account_service.get_account(access_token)
        image_items: list[dict[str, Any]] = []
        seen_image_urls: set[str] = set()
        emitted_output = False
        try:
            with GrokAppChatClient(access_token, account) as client:
                for event in client.stream_events(payload):
                    token, thinking = extract_app_chat_token(event)
                    if token and not thinking:
                        emitted_output = True
                        yield ImageOutput(kind="progress", model=spec.id, index=1, total=n, text=token, upstream_event_type="app_chat.token")
                    for image_url in extract_app_chat_image_urls(event):
                        if image_url in seen_image_urls:
                            continue
                        image_items.append({"url": image_url, "revised_prompt": prompt})
                        seen_image_urls.add(image_url)
                        if len(image_items) >= n:
                            break
                    moderation_message = app_chat_moderation_message(event)
                    if moderation_message:
                        raise ImageGenerationError(moderation_message, status_code=400, error_type="invalid_request_error", code="content_policy_violation")
                    if is_app_chat_final_event(event):
                        break
        except GrokConsoleError as exc:
            if emitted_output or not _is_retryable_app_chat_account_error(exc):
                raise _image_generation_error_from_grok(exc) from exc
            excluded_tokens.add(access_token)
            last_error = exc
            continue
        account_service.mark_text_used(access_token)
        image_items = image_items[:n]
        if image_items:
            yield ImageOutput(kind="result", model=spec.id, index=1, total=n, data=_format_grok_image_items(image_items, response_format))
            return
        raise ImageGenerationError("Grok image generation reached final response without an image", status_code=502, code="image_generation_failed")
    raise _no_available_grok_image_account_error(last_error)


def app_chat_image_edit_outputs(
    body: dict[str, Any],
    spec: ModelSpec,
    prompt: str,
    images: list[tuple[bytes, str, str]],
    n: int = 1,
    size: str | None = None,
) -> Iterator[ImageOutput]:
    validate_grok_image_edit_request(images, n, size)
    response_format = grok_image_response_format(body.get("response_format"))

    from services.account_service import account_service

    excluded_tokens: set[str] = set()
    last_error: GrokConsoleError | None = None
    for _ in range(_APP_CHAT_MAX_ACCOUNT_ATTEMPTS):
        access_token = account_service.get_grok_app_chat_access_token(spec, excluded_tokens=excluded_tokens)
        if not access_token:
            raise _no_available_grok_image_account_error(last_error)
        account = account_service.get_account(access_token)
        emitted_output = False
        try:
            with GrokAppChatClient(access_token, account) as client:
                references = [
                    client.upload_image_edit_reference(data, filename, mime_type)
                    for data, filename, mime_type in images
                ]
                edit_prompt = replace_grok_image_placeholders(prompt, references)
                parent_post_id, edit_prompt = client.create_image_edit_parent_post(edit_prompt)
                final_urls: dict[int, str] = {}
                user_id = _extract_x_user_id(access_token)
                for event in client.stream_image_edit_events(
                    edit_prompt,
                    [reference.content_url for reference in references],
                    parent_post_id,
                ):
                    stream = _extract_grok_streaming_image_response(event)
                    if stream:
                        try:
                            progress = int(stream.get("progress") or 0)
                        except (TypeError, ValueError):
                            progress = 0
                        emitted_output = True
                        yield ImageOutput(
                            kind="progress",
                            model=spec.id,
                            index=1,
                            total=n,
                            text=str(progress) if progress else "",
                            upstream_event_type="app_chat.image_edit_progress",
                        )
                    final_urls.update(extract_grok_image_edit_final_urls(event, user_id))
                    moderation_message = app_chat_moderation_message(event)
                    if moderation_message:
                        raise ImageGenerationError(moderation_message, status_code=400, error_type="invalid_request_error", code="content_policy_violation")
                    if is_app_chat_final_event(event):
                        break
        except GrokConsoleError as exc:
            if emitted_output or not _is_retryable_app_chat_account_error(exc):
                raise _image_generation_error_from_grok(exc) from exc
            excluded_tokens.add(access_token)
            last_error = exc
            continue
        account_service.mark_text_used(access_token)
        image_items = [
            {"url": url, "revised_prompt": prompt}
            for _, url in sorted(final_urls.items())[:n]
        ]
        if image_items:
            yield ImageOutput(kind="result", model=spec.id, index=1, total=n, data=_format_grok_image_items(image_items, response_format))
            return
        raise ImageGenerationError("Grok image edit did not return an image", status_code=502, code="image_edit_failed")
    raise _no_available_grok_image_account_error(last_error)


def console_chat_completion(body: dict[str, Any], spec: ModelSpec, messages: list[dict[str, Any]]) -> GrokConsoleCompletion:
    from services.account_service import account_service

    payload = build_console_payload(spec, body, messages)
    access_token = account_service.get_grok_console_access_token()
    if not access_token:
        raise HTTPException(status_code=503, detail={"error": "no available Grok account"})
    try:
        with GrokConsoleClient(access_token) as client:
            response_json = client.create_response(payload)
    except GrokConsoleError as exc:
        account_service.mark_grok_console_used(access_token, success=False)
        raise HTTPException(status_code=exc.status_code, detail=exc.to_http_detail()) from exc
    completion = extract_console_completion(response_json)
    if not completion.content and not completion.reasoning_content:
        account_service.mark_grok_console_used(access_token, success=False)
        raise HTTPException(status_code=502, detail={"error": "Grok upstream response did not contain text"})
    account_service.mark_grok_console_used(access_token, success=True)
    return completion


def console_chat_completion_events(body: dict[str, Any], spec: ModelSpec, messages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    from services.account_service import account_service

    payload = build_console_payload(spec, body, messages)
    access_token = account_service.get_grok_console_access_token()
    if not access_token:
        raise HTTPException(status_code=503, detail={"error": "no available Grok account"})
    try:
        with GrokConsoleClient(access_token) as client:
            for event in client.stream_response(payload):
                yield event
    except GrokConsoleError as exc:
        account_service.mark_grok_console_used(access_token, success=False)
        raise HTTPException(status_code=exc.status_code, detail=exc.to_http_detail()) from exc
    account_service.mark_grok_console_used(access_token, success=True)


def chat_completion(body: dict[str, Any], spec: ModelSpec, messages: list[dict[str, Any]]) -> str:
    return console_chat_completion(body, spec, messages).content
