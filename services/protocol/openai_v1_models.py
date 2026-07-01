from __future__ import annotations

from typing import Any

from services.providers.base import GPT_PROVIDER
from services.providers.gemini.models import gemini_model_metadata
from services.providers.gpt.models import GPT_IMAGE_MODEL_IDS, gpt_fallback_model_metadata, gpt_image_model_metadata
from services.providers.grok.models import grok_model_metadata


def _append_model(data: list[dict[str, Any]], seen: set[str], item: dict[str, Any]) -> None:
    model = str(item.get("id") or "").strip()
    if not model or model in seen:
        return
    seen.add(model)
    data.append(item)


def _with_provider(item: dict[str, Any], provider: str) -> dict[str, Any]:
    model = str(item.get("id") or "").strip()
    result = dict(item)
    result["provider"] = provider
    result.setdefault("root", model)
    result.setdefault("parent", None)
    result.setdefault("permission", [])
    return result


def _empty_model_result() -> dict[str, Any]:
    return {"object": "list", "data": []}


def _fetch_chatgpt_models(OpenAIBackendAPI: type, access_token: str = "", account_proxy: str = "") -> dict[str, Any]:
    with OpenAIBackendAPI(access_token, account_proxy=account_proxy) as backend:
        return backend.list_models()


def _get_gpt_access_token() -> str:
    try:
        from services.account_service import account_service
        return account_service.get_text_access_token(provider=GPT_PROVIDER)
    except Exception:
        return ""


def _get_account_proxy(token: str) -> str:
    if not token:
        return ""
    try:
        from services.account_service import account_service
        account = account_service.get_account(token)
        return str((account or {}).get("proxy") or "")
    except Exception:
        return ""


def list_models() -> dict[str, Any]:
    try:
        from services.openai_backend_api import OpenAIBackendAPI
    except Exception:
        result = _empty_model_result()
    else:
        access_token = _get_gpt_access_token()
        if access_token:
            try:
                result = _fetch_chatgpt_models(OpenAIBackendAPI, access_token, _get_account_proxy(access_token))
            except Exception:
                try:
                    result = _fetch_chatgpt_models(OpenAIBackendAPI)
                except Exception:
                    result = _empty_model_result()
        else:
            try:
                result = _fetch_chatgpt_models(OpenAIBackendAPI)
            except Exception:
                result = _empty_model_result()
    data = result.get("data")
    if not isinstance(data, list):
        result = _empty_model_result()
        data = result["data"]
    normalized_data: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in data:
        if isinstance(item, dict):
            _append_model(normalized_data, seen, _with_provider(item, GPT_PROVIDER))
    for item in gpt_fallback_model_metadata():
        _append_model(normalized_data, seen, item)
    for item in gpt_image_model_metadata():
        _append_model(normalized_data, seen, item)
    for item in grok_model_metadata():
        _append_model(normalized_data, seen, item)
    for item in gemini_model_metadata():
        _append_model(normalized_data, seen, item)
    result["data"] = normalized_data
    return result
