from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

EXPORT_FILENAME = "webchat2api_gemini.txt"
SECRET_KEYS = ("access_token", "accessToken", "cookies", "__Secure-1PSID", "__Secure-1PSIDTS", "SNlM0e", "session_token", "at")
SESSION_TOKEN_FIELDS = ("session_token", "SNlM0e", "at")
GEMINI_REQUIRED_COOKIES = ("__Secure-1PSID", "__Secure-1PSIDTS")
GEMINI_SENSITIVE_COOKIE_NAMES = ("__Secure-1PSID", "__Secure-1PSIDTS", "SNlM0e", "at", "session_token")
GEMINI_NON_COOKIE_FIELDS = ("SNlM0e", "session_token", "at")
UNAVAILABLE_STATUSES = {"禁用", "异常", "限流", "missing_gemini_session", "missing_psid"}
AUTH_FAILURE_MARKERS = (
    "auth",
    "login",
    "unauthorized",
    "forbidden",
    "snlm0e",
    "secure-1psid",
    "session expired",
    "invalid session",
    "expired session",
    "invalid token",
    "expired token",
    "missing token",
    "credential",
)


@dataclass(frozen=True)
class GeminiCookieState:
    cookie_header: str
    cookies: dict[str, str]
    psid: str
    psidts: str


@dataclass(frozen=True)
class GeminiRotateCookiesResult:
    cookie_header: str
    cookies: dict[str, str]
    psidts: str
    refreshed_psidts: str


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().strip('"').strip("'").rstrip(";").strip()


def parse_cookie_header(cookie_header: str) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for part in str(cookie_header or "").split(";"):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        name = name.strip()
        value = clean_string(value)
        if name:
            cookies[name] = value
    return cookies


def _cookie_field(item: dict[str, Any], name: str) -> str:
    cookies = item.get("cookies")
    if isinstance(cookies, dict):
        value = clean_string(cookies.get(name))
        if value:
            return value
    return clean_string(item.get(name))


def cookie_header(psid: str, psidts: str) -> str:
    parts = []
    if psid:
        parts.append(f"__Secure-1PSID={psid}")
    if psidts:
        parts.append(f"__Secure-1PSIDTS={psidts}")
    return "; ".join(parts)


def _merged_cookie_fields(item: dict[str, Any]) -> dict[str, str]:
    cookies = parse_cookie_header(clean_string(item.get("access_token") or item.get("accessToken")))
    stored_cookies = item.get("cookies")
    if isinstance(stored_cookies, dict):
        for name, value in stored_cookies.items():
            value_text = clean_string(value)
            if value_text:
                cookies[str(name)] = value_text
    return cookies


def cookie_header_from_mapping(cookies: dict[str, Any]) -> str:
    parts: list[str] = []
    for name, value in cookies.items():
        name_text = str(name or "").strip()
        value_text = clean_string(value)
        if name_text and value_text:
            parts.append(f"{name_text}={value_text}")
    return "; ".join(parts)


def sanitize_cookie_header(cookie_header: str) -> str:
    cookies = parse_cookie_header(cookie_header)
    for name in list(cookies):
        if name in GEMINI_SENSITIVE_COOKIE_NAMES:
            cookies[name] = "[redacted]"
    return cookie_header_from_mapping(cookies)


def merge_cookie_headers(*cookie_headers: str) -> str:
    cookies: dict[str, str] = {}
    for cookie_header_value in cookie_headers:
        for name, value in parse_cookie_header(cookie_header_value).items():
            if value:
                cookies[name] = value
    return cookie_header_from_mapping(cookies)


def cookie_header_from_response(response: object) -> str:
    cookies_attr = getattr(response, "cookies", None)
    response_cookies = cookies_attr() if callable(cookies_attr) else cookies_attr
    parts: list[str] = []
    if isinstance(response_cookies, dict):
        parts.extend(f"{name}={clean_string(value)}" for name, value in response_cookies.items())
    elif isinstance(response_cookies, Iterable) and not isinstance(response_cookies, (str, bytes)):
        for cookie in response_cookies:
            name = getattr(cookie, "name", None)
            value = getattr(cookie, "value", None)
            if name is not None and value is not None:
                parts.append(f"{name}={clean_string(value)}")
    headers = getattr(response, "headers", None)
    raw_set_cookie = ""
    if isinstance(headers, dict):
        raw_set_cookie = str(headers.get("set-cookie") or headers.get("Set-Cookie") or "")
    if raw_set_cookie:
        for item in re.split(r",\s*(?=[^;,\s]+=)", raw_set_cookie):
            first = item.strip().split(";", 1)[0]
            if "=" in first:
                parts.append(first)
    return merge_cookie_headers(*parts)


def merge_response_cookies(cookie_header_value: str, response: object) -> str:
    return merge_cookie_headers(cookie_header_value, cookie_header_from_response(response))



def gemini_session_token(item: dict[str, Any]) -> str:
    for name in SESSION_TOKEN_FIELDS:
        value = clean_string(item.get(name))
        if value:
            return value
    access_cookies = parse_cookie_header(clean_string(item.get("access_token") or item.get("accessToken")))
    stored_cookies = item.get("cookies")
    stored_cookie_values = stored_cookies if isinstance(stored_cookies, dict) else {}
    for name in ("SNlM0e", "at"):
        value = clean_string(access_cookies.get(name)) or clean_string(stored_cookie_values.get(name))
        if value:
            return value
    return ""


def gemini_cookie_state(item: dict[str, Any], require_session_cookies: bool = False) -> GeminiCookieState:
    raw_cookies = _merged_cookie_fields(item)
    psid = clean_string(raw_cookies.get("__Secure-1PSID")) or _cookie_field(item, "__Secure-1PSID")
    psidts = clean_string(raw_cookies.get("__Secure-1PSIDTS")) or _cookie_field(item, "__Secure-1PSIDTS")
    if psid:
        raw_cookies["__Secure-1PSID"] = psid
    if psidts:
        raw_cookies["__Secure-1PSIDTS"] = psidts
    for name in GEMINI_NON_COOKIE_FIELDS:
        raw_cookies.pop(name, None)
    if require_session_cookies:
        missing = [name for name in GEMINI_REQUIRED_COOKIES if not raw_cookies.get(name)]
        if missing:
            raise ValueError(f"Gemini account is missing required cookie(s): {', '.join(missing)}")
    cookies = {name: value for name, value in raw_cookies.items() if value}
    return GeminiCookieState(
        cookie_header=cookie_header_from_mapping(cookies),
        cookies=cookies,
        psid=cookies.get("__Secure-1PSID", ""),
        psidts=cookies.get("__Secure-1PSIDTS", ""),
    )


def gemini_rotate_cookies_result(cookie_header_value: str, response: object) -> GeminiRotateCookiesResult:
    previous_cookies = parse_cookie_header(cookie_header_value)
    merged_header = merge_response_cookies(cookie_header_value, response)
    merged_cookies = parse_cookie_header(merged_header)
    psidts = clean_string(merged_cookies.get("__Secure-1PSIDTS"))
    previous_psidts = clean_string(previous_cookies.get("__Secure-1PSIDTS"))
    refreshed_psidts = psidts if psidts and psidts != previous_psidts else ""
    return GeminiRotateCookiesResult(
        cookie_header=merged_header,
        cookies=merged_cookies,
        psidts=psidts,
        refreshed_psidts=refreshed_psidts,
    )


def account_category(item: dict[str, Any]) -> str:
    access_token, psid, psidts = normalize_account_credentials(dict(item))
    session_token = gemini_session_token(item)
    has_cookie_header = bool(parse_cookie_header(access_token))
    if psid and psidts and session_token:
        return "full_session"
    if psid and psidts:
        return "psid_psidts"
    if psid:
        return "psid_only"
    if session_token or has_cookie_header:
        return "session_token_only"
    return "missing_session"


def account_status(item: dict[str, Any]) -> str:
    category = account_category(item)
    if category == "missing_session":
        return "missing_gemini_session"
    if category == "session_token_only":
        return "missing_psid"
    return "usable_gemini_session"


def has_gemini_session(item: dict[str, Any]) -> bool:
    return account_category(item) != "missing_session"


def normalize_account_credentials(item: dict[str, Any]) -> tuple[str, str, str]:
    raw_cookies = _merged_cookie_fields(item)
    psid = clean_string(raw_cookies.get("__Secure-1PSID")) or _cookie_field(item, "__Secure-1PSID")
    psidts = clean_string(raw_cookies.get("__Secure-1PSIDTS")) or _cookie_field(item, "__Secure-1PSIDTS")
    access_token = clean_string(item.get("access_token") or item.get("accessToken"))
    if not access_token and psid:
        access_token = cookie_header(psid, psidts)
    return access_token, psid, psidts


def normalize_access_token(item: dict[str, Any]) -> str:
    return normalize_account_credentials(item)[0]


def normalize_account(account: dict[str, Any]) -> dict[str, Any]:
    access_token, psid, psidts = normalize_account_credentials(account)
    account["access_token"] = access_token
    account["__Secure-1PSID"] = psid
    account["__Secure-1PSIDTS"] = psidts
    cookies = _merged_cookie_fields(account)
    if psid:
        cookies["__Secure-1PSID"] = psid
    if psidts:
        cookies["__Secure-1PSIDTS"] = psidts
    for name in SESSION_TOKEN_FIELDS:
        value = clean_string(account.get(name)) or clean_string(cookies.get(name))
        if value:
            account[name] = value
    account["cookies"] = {name: value for name, value in cookies.items() if value}
    account["user_agent"] = clean_string(account.get("user_agent")) or None
    category = account_category(account)
    account["account_category"] = category
    account["account_status"] = account_status(account)
    account["has_gemini_session"] = category != "missing_session"
    return account


def delete_token_matches_account(token: str, account: dict[str, Any]) -> bool:
    token_access, token_psid, token_psidts = normalize_account_credentials({"access_token": token})
    account_access, account_psid, account_psidts = normalize_account_credentials(dict(account))
    return bool(
        (token_access and token_access == account_access)
        or (token_psid and token_psid == account_psid)
        or (token_psidts and token_psidts == account_psidts)
    )


def supports_refresh(account: dict[str, Any]) -> bool:
    return account_category(account) in {"full_session", "psid_psidts"}


def validate_remote_info(access_token: str, account: dict[str, Any] | None = None) -> dict[str, Any]:
    from services.providers.gemini.client import GeminiWebClient, account_cookie_header, gemini_session_writeback

    source = dict(account or {})
    if access_token:
        source.setdefault("access_token", access_token)
    cookie_header_value = account_cookie_header(source)
    with GeminiWebClient(cookie_header_value, source.get("user_agent"), account=source) as client:
        client.rotate_psidts()
        session_token = client.bootstrap_session_token()
        return gemini_session_writeback(source, client.cookie_header, session_token)


def remote_error_status(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


def refresh_error_message(exc: Exception) -> str:
    message = clean_string(exc)
    return message or "Gemini session refresh failed"


def is_auth_failure_payload(payload: Any) -> bool:
    if isinstance(payload, dict):
        status = payload.get("status") or payload.get("status_code") or payload.get("code")
        status_text = clean_string(status)
        try:
            if status_text and int(status_text) in {401, 403}:
                return True
        except (TypeError, ValueError):
            pass
        for key in ("error", "message", "detail", "code", "reason", "error_description"):
            text = clean_string(payload.get(key)).lower()
            if any(marker in text for marker in AUTH_FAILURE_MARKERS):
                return True
        return any(is_auth_failure_payload(value) for value in payload.values())
    if isinstance(payload, (list, tuple, set)):
        return any(is_auth_failure_payload(value) for value in payload)
    return False


def is_image_account_available(account: dict[str, Any]) -> bool:
    return False


def normalize_console_quota(value: Any) -> dict[str, Any]:
    return {}


def reset_console_quota_if_ready(account: dict[str, Any], current_time: float) -> dict[str, Any]:
    return dict(account)


def is_console_account_available(account: dict[str, Any], current_time: float) -> bool:
    return False


def requested_tiers(spec: Any) -> list[str]:
    return []


def account_has_capability(account: dict[str, Any], spec: Any) -> bool:
    return account_category(account) != "missing_session"


def tier_matches(account_tier: str, requested_tier: str) -> bool:
    return False


def normalize_tier(value: Any) -> str:
    return clean_string(value).lower()


def export_filename() -> str:
    return EXPORT_FILENAME


def build_export_item(account: dict[str, Any]) -> dict[str, str] | None:
    access_token = clean_string(account.get("access_token"))
    if not access_token:
        return None
    return {
        "type": clean_string(account.get("export_type")) or "codex",
        "email": clean_string(account.get("email")),
        "expired": clean_string(account.get("expired")),
        "id_token": clean_string(account.get("id_token")),
        "account_id": clean_string(account.get("account_id")),
        "access_token": access_token,
        "sso": clean_string(account.get("sso")),
        "last_refresh": clean_string(account.get("last_refresh")),
        "refresh_token": clean_string(account.get("refresh_token")),
    }


def account_row_id(account: dict[str, Any]) -> str:
    access_token, psid, psidts = normalize_account_credentials(dict(account))
    source = "\0".join([access_token, psid, psidts, clean_string(account.get("account_id"))])
    if not source.strip("\0"):
        return ""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def sanitize_account(item: dict[str, Any]) -> dict[str, Any]:
    account = dict(item)
    row_id = account_row_id(item)
    for key in SECRET_KEYS:
        account.pop(key, None)
    if row_id:
        account["row_id"] = row_id
    category = account_category(item)
    account["has_gemini_session"] = category != "missing_session"
    account["account_category"] = category
    account["account_status"] = account_status(item)
    return account
