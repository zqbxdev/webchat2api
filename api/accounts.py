from __future__ import annotations

from typing import Any, Literal, Sequence

from fastapi import APIRouter, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel, Field

from services.auth_service import auth_service

from api.support import (
    require_admin,
    sanitize_cpa_pool,
    sanitize_cpa_pools,
    sanitize_remote_account_source,
    sanitize_remote_account_sources,
    sanitize_sub2api_server,
    sanitize_sub2api_servers,
)
from services.account_service import account_service
from services.providers.base import GEMINI_PROVIDER, GROK_PROVIDER
from services.providers.registry import normalize_account_provider, normalize_provider
from services.providers.registry import account_strategy
from services.cpa_service import cpa_config, cpa_import_service, list_remote_files
from services.remote_account_service import REMOTE_ACCOUNT_SYNC_FAILED, remote_account_config, remote_account_import_service
from services.sub2api_service import (
    list_remote_accounts as sub2api_list_remote_accounts,
    list_remote_groups as sub2api_list_remote_groups,
    sub2api_config,
    sub2api_import_service,
)



class UserKeyCreateRequest(BaseModel):
    name: str = ""


class UserKeyUpdateRequest(BaseModel):
    name: str | None = None
    enabled: bool | None = None
    key: str | None = None


class AccountCreateRequest(BaseModel):
    tokens: list[str] = Field(default_factory=list)
    accounts: list[dict[str, Any]] = Field(default_factory=list)
    provider: Literal["gpt", "grok", "gemini"] | None = None


class AccountDeleteIdentifier(BaseModel):
    account_id: str | None = None
    row_id: str | None = None


class AccountDeleteRequest(BaseModel):
    tokens: list[str] = Field(default_factory=list)
    identifiers: list[AccountDeleteIdentifier] = Field(default_factory=list)
    mode: Literal["tokens", "limited"] = "tokens"
    provider: Literal["gpt", "grok", "gemini"] | None = None


class AccountRefreshRequest(BaseModel):
    access_tokens: list[str] = Field(default_factory=list)
    identifiers: list[AccountDeleteIdentifier] = Field(default_factory=list)
    provider: Literal["gpt", "grok", "gemini"] | None = None


class AccountValidateRequest(BaseModel):
    access_tokens: list[str] = Field(default_factory=list)
    identifiers: list[AccountDeleteIdentifier] = Field(default_factory=list)
    provider: Literal["gpt", "grok", "gemini"] | None = None


class AccountExportRequest(BaseModel):
    access_tokens: list[str] = Field(default_factory=list)
    identifiers: list[AccountDeleteIdentifier] = Field(default_factory=list)
    provider: Literal["gpt", "grok", "gemini"]


class AccountUpdateRequest(BaseModel):
    access_token: str = ""
    account_id: str | None = None
    row_id: str | None = None
    type: str | None = None
    provider: str | None = None
    target_provider: Literal["gpt", "grok", "gemini"] | None = None
    status: str | None = None
    quota: int | None = None
    proxy: str | None = None


class CPAPoolCreateRequest(BaseModel):
    name: str = ""
    base_url: str = ""
    secret_key: str = ""


class CPAPoolUpdateRequest(BaseModel):
    name: str | None = None
    base_url: str | None = None
    secret_key: str | None = None


class CPAImportRequest(BaseModel):
    names: list[str] = Field(default_factory=list)


class Sub2APIServerCreateRequest(BaseModel):
    name: str = ""
    base_url: str = ""
    email: str = ""
    password: str = ""
    api_key: str = ""
    group_id: str = ""


class Sub2APIServerUpdateRequest(BaseModel):
    name: str | None = None
    base_url: str | None = None
    email: str | None = None
    password: str | None = None
    api_key: str | None = None
    group_id: str | None = None


class Sub2APIImportRequest(BaseModel):
    account_ids: list[str] = Field(default_factory=list)


class RemoteAccountSourceCreateRequest(BaseModel):
    name: str = ""
    enabled: bool = True
    url: str = ""
    method: Literal["GET", "POST"] = "GET"
    auth_header: str = ""
    auth_token: str = ""
    bearer_token: str = ""
    provider: Literal["", "gpt", "grok", "gemini"] = ""
    sync_strategy: Literal["merge", "replace"] = "merge"
    interval_seconds: int | None = None


class RemoteAccountSourceUpdateRequest(BaseModel):
    name: str | None = None
    enabled: bool | None = None
    url: str | None = None
    method: Literal["GET", "POST"] | None = None
    auth_header: str | None = None
    auth_token: str | None = None
    bearer_token: str | None = None
    provider: Literal["", "gpt", "grok", "gemini"] | None = None
    sync_strategy: Literal["merge", "replace"] | None = None
    interval_seconds: int | None = None


class RemoteAccountInjectRequest(BaseModel):
    payload: Any | None = None
    accounts: list[Any] | None = None
    tokens: list[str] | None = None
    strategy: Literal["merge", "replace"] = "merge"
    source_id: str = ""
    source_name: str = ""
    provider: Literal["gpt", "grok", "gemini"] = "gpt"


def _account_payload_provider(item: dict[str, Any], provider: str | None) -> str:
    provider_value = item.get("provider") or provider
    if str(provider_value or "").strip():
        return normalize_account_provider(provider_value)
    return normalize_provider(provider_value)


def _account_payload_token(item: dict[str, Any], provider: str | None = None) -> str:
    try:
        payload_provider = _account_payload_provider(item, provider)
    except ValueError:
        return ""
    payload = dict(item)
    payload.setdefault("provider", payload_provider)
    return str(_account_strategy(payload_provider).normalize_access_token(payload) or "").strip()


def _unique_tokens(tokens: list[str]) -> list[str]:
    return list(dict.fromkeys(str(token or "").strip() for token in tokens if str(token or "").strip()))


def _should_refresh_created_accounts(provider: str | None) -> bool:
    return normalize_provider(provider) != GROK_PROVIDER


def _delete_identifiers(identifiers: Sequence[AccountDeleteIdentifier | dict[str, Any]]) -> list[dict[str, str]]:
    payloads: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for identifier in identifiers:
        payload: dict[str, str] = {}
        if isinstance(identifier, dict):
            raw_account_id = identifier.get("account_id")
            raw_row_id = identifier.get("row_id")
        else:
            raw_account_id = identifier.account_id
            raw_row_id = identifier.row_id
        account_id = str(raw_account_id or "").strip()
        row_id = str(raw_row_id or "").strip()
        if account_id:
            payload["account_id"] = account_id
        if row_id:
            payload["row_id"] = row_id
        if not payload:
            continue
        key = (payload.get("account_id", ""), payload.get("row_id", ""))
        if key in seen:
            continue
        payloads.append(payload)
        seen.add(key)
    return payloads


def _account_strategy(provider: Any):
    return account_strategy(normalize_account_provider(provider))


def _gemini_import_requested(provider: str | None, payloads: list[dict[str, Any]]) -> bool:
    if normalize_provider(provider) == GEMINI_PROVIDER:
        return True
    for item in payloads:
        provider_value = str(item.get("provider") or "").strip()
        if not provider_value:
            continue
        try:
            if normalize_account_provider(provider_value) == GEMINI_PROVIDER:
                return True
        except ValueError:
            continue
    return False


def _validate_gemini_import_payloads(tokens: list[str], payloads: list[dict[str, Any]], provider: str | None) -> None:
    if not _gemini_import_requested(provider, payloads):
        return
    if tokens:
        raise HTTPException(status_code=400, detail={"error": "Gemini 目前只支持 Cookie 双字段导入，请分别填写 __Secure-1PSID 和 __Secure-1PSIDTS 的值"})
    if not payloads:
        raise HTTPException(status_code=400, detail={"error": "Gemini 目前只支持 Cookie 双字段导入"})
    top_level_gemini = normalize_provider(provider) == GEMINI_PROVIDER
    allowed_keys = {"provider", "__Secure-1PSID", "__Secure-1PSIDTS", "proxy"}
    for item in payloads:
        item_provider = str(item.get("provider") or "").strip()
        if top_level_gemini and item_provider and normalize_account_provider(item_provider) != GEMINI_PROVIDER:
            raise HTTPException(status_code=400, detail={"error": "Gemini 导入不能混用其他供应商账号"})
        extra_keys = {key for key, value in item.items() if value is not None} - allowed_keys
        if extra_keys:
            raise HTTPException(status_code=400, detail={"error": "Gemini 目前只支持 Cookie 双字段导入，不支持完整 Cookie、access_token、JSON 或其他导入方式"})
        psid = str(item.get("__Secure-1PSID") or "").strip()
        psidts = str(item.get("__Secure-1PSIDTS") or "").strip()
        if not psid or not psidts:
            raise HTTPException(status_code=400, detail={"error": "请分别填写 __Secure-1PSID 和 __Secure-1PSIDTS 的值"})
        if "=" in psid or ";" in psid or "=" in psidts or ";" in psidts:
            raise HTTPException(status_code=400, detail={"error": "Gemini Cookie 双字段只接受等号右侧的值，不要包含 cookie 名称、等号或分号"})


def _normalize_grok_sso_import_token(value: str) -> str:
    token = str(value or "").strip()
    if not token or ";" in token:
        return ""
    name, separator, cookie_value = token.partition("=")
    if separator:
        return cookie_value.strip() if name.strip().lower() == "sso" and cookie_value.strip() else ""
    return token


def _grok_import_requested(provider: str | None, payloads: list[dict[str, Any]]) -> bool:
    if normalize_provider(provider) == GROK_PROVIDER:
        return True
    for item in payloads:
        provider_value = str(item.get("provider") or "").strip()
        if not provider_value:
            continue
        try:
            if normalize_account_provider(provider_value) == GROK_PROVIDER:
                return True
        except ValueError:
            continue
    return False


def _validate_grok_import_payloads(tokens: list[str], payloads: list[dict[str, Any]], provider: str | None) -> list[str]:
    if not _grok_import_requested(provider, payloads):
        return tokens
    if not tokens and not payloads:
        raise HTTPException(status_code=400, detail={"error": "Grok 导入只接受裸 SSO 值，或每行一个 sso=<值>"})
    allowed_keys = {"provider", "sso", "proxy"}
    for item in payloads:
        item_provider = str(item.get("provider") or provider or "").strip()
        if item_provider and normalize_account_provider(item_provider) != GROK_PROVIDER:
            raise HTTPException(status_code=400, detail={"error": "Grok 导入不能混用其他供应商账号"})
        extra_keys = {key for key, value in item.items() if value is not None} - allowed_keys
        sso = str(item.get("sso") or "").strip()
        if extra_keys or not _normalize_grok_sso_import_token(sso):
            raise HTTPException(status_code=400, detail={"error": "Grok 导入只接受裸 SSO 值，或每行一个 sso=<值>；不支持 sso-rw、完整 Cookie header、其他 Cookie 名称、JSON、CPA 或 cookies"})
    normalized_tokens = [_normalize_grok_sso_import_token(token) for token in tokens]
    if not all(normalized_tokens):
        raise HTTPException(status_code=400, detail={"error": "Grok 导入只接受裸 SSO 值，或每行一个 sso=<值>；不支持 sso-rw、完整 Cookie header、其他 Cookie 名称、JSON、CPA 或 cookies"})
    return normalized_tokens


def _export_filename(provider: Literal["gpt", "grok", "gemini"]) -> str:
    return _account_strategy(provider).export_filename()


def sanitize_account(item: dict[str, Any]) -> dict[str, Any]:
    return _account_strategy(item.get("provider")).sanitize_account(item)


def sanitize_accounts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [sanitize_account(item) for item in items]


def sanitize_account_result(result: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(result)
    if isinstance(sanitized.get("items"), list):
        sanitized["items"] = sanitize_accounts([item for item in sanitized["items"] if isinstance(item, dict)])
    if isinstance(sanitized.get("item"), dict):
        sanitized["item"] = sanitize_account(sanitized["item"])
    return sanitized


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/auth/users")
    async def list_user_keys(authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return {"items": auth_service.list_keys(role="user")}

    @router.post("/api/auth/users")
    async def create_user_key(body: UserKeyCreateRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        try:
            item, raw_key = auth_service.create_key(role="user", name=body.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        return {"item": item, "key": raw_key, "items": auth_service.list_keys(role="user")}

    @router.post("/api/auth/users/{key_id}")
    async def update_user_key(
            key_id: str,
            body: UserKeyUpdateRequest,
            authorization: str | None = Header(default=None),
    ):
        require_admin(authorization)
        updates = {
            key: value
            for key, value in {
                "name": body.name,
                "enabled": body.enabled,
                "key": body.key,
            }.items()
            if value is not None
        }
        if not updates:
            raise HTTPException(status_code=400, detail={"error": "还没有检测到改动，请修改后再保存"})
        try:
            item = auth_service.update_key(key_id, updates, role="user")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        if item is None:
            raise HTTPException(status_code=404, detail={"error": "这条用户密钥不存在，可能已经被删除"})
        return {"item": item, "items": auth_service.list_keys(role="user")}

    @router.delete("/api/auth/users/{key_id}")
    async def delete_user_key(key_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        if not auth_service.delete_key(key_id, role="user"):
            raise HTTPException(status_code=404, detail={"error": "这条用户密钥不存在，可能已经被删除"})
        return {"items": auth_service.list_keys(role="user")}

    @router.get("/api/accounts")
    async def get_accounts(provider: Literal["gpt", "grok", "gemini"] | None = None, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return {"items": sanitize_accounts(account_service.list_accounts(provider=provider))}

    @router.post("/api/accounts")
    async def create_accounts(body: AccountCreateRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        account_payloads = [item for item in body.accounts if isinstance(item, dict)]
        raw_tokens = [str(token or "").strip() for token in body.tokens if str(token or "").strip()]
        _validate_gemini_import_payloads(raw_tokens, account_payloads, body.provider)
        body_tokens = _validate_grok_import_payloads(raw_tokens, account_payloads, body.provider)
        scoped_payloads = [{**item, "provider": item.get("provider") or body.provider} for item in account_payloads]
        payload_tokens = [_account_payload_token(item, body.provider) for item in scoped_payloads]
        tokens = _unique_tokens([*body_tokens, *payload_tokens])
        if not tokens:
            raise HTTPException(status_code=400, detail={"error": "tokens is required"})
        if scoped_payloads:
            result = account_service.add_account_items(scoped_payloads)
            saved_tokens = {
                str(item.get("access_token") or "").strip()
                for item in result.get("items", [])
                if isinstance(item, dict) and str(item.get("access_token") or "").strip()
            }
            payload_token_set = set(_unique_tokens(payload_tokens))
            extra_tokens = [token for token in tokens if token not in payload_token_set]
            if extra_tokens:
                extra_result = account_service.add_accounts(extra_tokens, provider=body.provider)
                result["added"] = int(result.get("added") or 0) + int(extra_result.get("added") or 0)
                result["skipped"] = int(result.get("skipped") or 0) + int(extra_result.get("skipped") or 0)
                for item in extra_result.get("items", []):
                    if isinstance(item, dict) and (token := str(item.get("access_token") or "").strip()):
                        saved_tokens.add(token)
            refresh_tokens = [token for token in tokens if token in saved_tokens]
        else:
            result = account_service.add_accounts(tokens, provider=body.provider)
            saved_tokens = {
                str(item.get("access_token") or "").strip()
                for item in result.get("items", [])
                if isinstance(item, dict) and str(item.get("access_token") or "").strip()
            }
            refresh_tokens = [token for token in tokens if token in saved_tokens]
            if not refresh_tokens and not int(result.get("added") or 0) and not int(result.get("skipped") or 0):
                raise HTTPException(status_code=400, detail={"error": "tokens is required"})
        if _should_refresh_created_accounts(body.provider):
            refresh_result = account_service.refresh_accounts(refresh_tokens, provider=body.provider)
        else:
            refresh_result = {"refreshed": 0, "errors": [], "items": result.get("items", [])}
        return sanitize_account_result({
            **result,
            "refreshed": refresh_result.get("refreshed", 0),
            "errors": refresh_result.get("errors", []),
            "items": refresh_result.get("items", result.get("items", [])),
        })

    @router.delete("/api/accounts")
    async def delete_accounts(body: AccountDeleteRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        if body.mode == "limited":
            return sanitize_account_result(account_service.delete_limited_accounts(provider=body.provider))
        tokens = [str(token or "").strip() for token in body.tokens if str(token or "").strip()]
        identifiers = _delete_identifiers(body.identifiers)
        if not tokens and not identifiers:
            raise HTTPException(status_code=400, detail={"error": "tokens or identifiers is required"})
        return sanitize_account_result(account_service.delete_accounts(tokens, provider=body.provider, identifiers=identifiers))

    @router.post("/api/accounts/refresh")
    async def refresh_accounts(body: AccountRefreshRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        access_tokens = [str(token or "").strip() for token in body.access_tokens if str(token or "").strip()]
        identifiers = _delete_identifiers(body.identifiers)
        if not access_tokens and not identifiers:
            access_tokens = account_service.list_tokens(provider=body.provider)
        if not access_tokens and not identifiers:
            raise HTTPException(status_code=400, detail={"error": "access_tokens or identifiers is required"})
        return sanitize_account_result(account_service.refresh_accounts(access_tokens, provider=body.provider, identifiers=identifiers))

    @router.post("/api/accounts/validate")
    async def validate_accounts(body: AccountValidateRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        provider = normalize_account_provider(body.provider or GROK_PROVIDER)
        if provider != GROK_PROVIDER:
            raise HTTPException(status_code=400, detail={"error": "unsupported provider for account validation"})
        access_tokens = [str(token or "").strip() for token in body.access_tokens if str(token or "").strip()]
        identifiers = _delete_identifiers(body.identifiers)
        if not access_tokens and not identifiers:
            access_tokens = account_service.list_tokens(provider=provider)
        if not access_tokens and not identifiers:
            raise HTTPException(status_code=400, detail={"error": "access_tokens or identifiers is required"})
        return sanitize_account_result(account_service.validate_accounts(access_tokens, provider=provider, identifiers=identifiers))

    @router.post("/api/accounts/export")
    async def export_accounts(body: AccountExportRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        access_tokens = _unique_tokens(body.access_tokens)
        identifiers = _delete_identifiers(body.identifiers)
        items = account_service.build_export_items(access_tokens, provider=body.provider, identifiers=identifiers)
        if not items:
            raise HTTPException(
                status_code=400,
                detail={"error": "没有可导出的账号，请检查 access_tokens 是否存在"},
            )

        filename = _export_filename(body.provider)
        return Response(
            account_service.build_export_text(items),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @router.post("/api/accounts/update")
    async def update_account(body: AccountUpdateRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        access_token = str(body.access_token or "").strip()
        identifier = _delete_identifiers([{"account_id": body.account_id, "row_id": body.row_id}])
        if not access_token and not identifier:
            raise HTTPException(status_code=400, detail={"error": "access_token or identifier is required"})
        updates = {
            key: value
            for key, value in {
                "type": body.type,
                "provider": body.provider,
                "status": body.status,
                "quota": body.quota,
                "proxy": body.proxy,
            }.items()
            if value is not None
        }
        if not updates:
            raise HTTPException(status_code=400, detail={"error": "还没有检测到改动，请修改后再保存"})
        if access_token:
            account = account_service.update_account(access_token, updates, provider=body.target_provider)
        else:
            account = account_service.update_account_by_identifier(identifier[0], updates, provider=body.target_provider)
        if account is None:
            raise HTTPException(status_code=404, detail={"error": "account not found"})
        return {"item": sanitize_account(account), "items": sanitize_accounts(account_service.list_accounts(provider=body.target_provider))}

    @router.get("/api/cpa/pools")
    async def list_cpa_pools(authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return {"pools": sanitize_cpa_pools(cpa_config.list_pools())}

    @router.post("/api/cpa/pools")
    async def create_cpa_pool(body: CPAPoolCreateRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        if not body.base_url.strip():
            raise HTTPException(status_code=400, detail={"error": "base_url is required"})
        if not body.secret_key.strip():
            raise HTTPException(status_code=400, detail={"error": "secret_key is required"})
        pool = cpa_config.add_pool(name=body.name, base_url=body.base_url, secret_key=body.secret_key)
        return {"pool": sanitize_cpa_pool(pool), "pools": sanitize_cpa_pools(cpa_config.list_pools())}

    @router.post("/api/cpa/pools/{pool_id}")
    async def update_cpa_pool(pool_id: str, body: CPAPoolUpdateRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        pool = cpa_config.update_pool(pool_id, body.model_dump(exclude_none=True))
        if pool is None:
            raise HTTPException(status_code=404, detail={"error": "pool not found"})
        return {"pool": sanitize_cpa_pool(pool), "pools": sanitize_cpa_pools(cpa_config.list_pools())}

    @router.delete("/api/cpa/pools/{pool_id}")
    async def delete_cpa_pool(pool_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        if not cpa_config.delete_pool(pool_id):
            raise HTTPException(status_code=404, detail={"error": "pool not found"})
        return {"pools": sanitize_cpa_pools(cpa_config.list_pools())}

    @router.get("/api/cpa/pools/{pool_id}/files")
    async def cpa_pool_files(pool_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        pool = cpa_config.get_pool(pool_id)
        if pool is None:
            raise HTTPException(status_code=404, detail={"error": "pool not found"})
        return {"pool_id": pool_id, "files": await run_in_threadpool(list_remote_files, pool)}

    @router.post("/api/cpa/pools/{pool_id}/import")
    async def cpa_pool_import(pool_id: str, body: CPAImportRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        pool = cpa_config.get_pool(pool_id)
        if pool is None:
            raise HTTPException(status_code=404, detail={"error": "pool not found"})
        try:
            job = cpa_import_service.start_import(pool, body.names)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        return {"import_job": job}

    @router.get("/api/cpa/pools/{pool_id}/import")
    async def cpa_pool_import_progress(pool_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        pool = cpa_config.get_pool(pool_id)
        if pool is None:
            raise HTTPException(status_code=404, detail={"error": "pool not found"})
        return {"import_job": pool.get("import_job")}


    @router.get("/api/remote-account/sources")
    async def list_remote_account_sources(authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return {"sources": sanitize_remote_account_sources(remote_account_config.list_sources())}

    @router.post("/api/remote-account/sources")
    async def create_remote_account_source(body: RemoteAccountSourceCreateRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        if not body.url.strip():
            raise HTTPException(status_code=400, detail={"error": "url is required"})
        try:
            source = remote_account_config.add_source(**body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        return {"source": sanitize_remote_account_source(source), "sources": sanitize_remote_account_sources(remote_account_config.list_sources())}

    @router.post("/api/remote-account/sources/{source_id}")
    async def update_remote_account_source(source_id: str, body: RemoteAccountSourceUpdateRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        try:
            source = remote_account_config.update_source(source_id, body.model_dump(exclude_none=True))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        if source is None:
            raise HTTPException(status_code=404, detail={"error": "source not found"})
        return {"source": sanitize_remote_account_source(source), "sources": sanitize_remote_account_sources(remote_account_config.list_sources())}

    @router.delete("/api/remote-account/sources/{source_id}")
    async def delete_remote_account_source(source_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        if not remote_account_config.delete_source(source_id):
            raise HTTPException(status_code=404, detail={"error": "source not found"})
        return {"sources": sanitize_remote_account_sources(remote_account_config.list_sources())}

    @router.post("/api/remote-account/sources/{source_id}/sync")
    async def sync_remote_account_source(source_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        source = remote_account_config.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail={"error": "source not found"})
        try:
            job = await run_in_threadpool(remote_account_import_service.sync_source, source, remote_account_config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": REMOTE_ACCOUNT_SYNC_FAILED}) from exc
        return {"import_job": job, "source": sanitize_remote_account_source(remote_account_config.get_source(source_id))}

    @router.get("/api/remote-account/sources/{source_id}/sync")
    async def remote_account_source_sync_progress(source_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        source = remote_account_config.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail={"error": "source not found"})
        return {"import_job": source.get("import_job")}

    @router.post("/api/remote-account/inject")
    async def inject_remote_accounts(body: RemoteAccountInjectRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        payload = body.payload
        if payload is None:
            if body.accounts is not None:
                payload = {"accounts": body.accounts}
            elif body.tokens is not None:
                payload = {"tokens": body.tokens}
        if payload is None:
            raise HTTPException(status_code=400, detail={"error": "payload, accounts, or tokens is required"})
        try:
            result = remote_account_import_service.inject_payload(
                payload,
                strategy=body.strategy,
                source_id=body.source_id,
                source_name=body.source_name,
                provider_default=body.provider,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        return result

    @router.get("/api/sub2api/servers")
    async def list_sub2api_servers(authorization: str | None = Header(default=None)):
        require_admin(authorization)
        return {"servers": sanitize_sub2api_servers(sub2api_config.list_servers())}

    @router.post("/api/sub2api/servers")
    async def create_sub2api_server(body: Sub2APIServerCreateRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        if not body.base_url.strip():
            raise HTTPException(status_code=400, detail={"error": "base_url is required"})
        has_login = body.email.strip() and body.password.strip()
        has_api_key = bool(body.api_key.strip())
        if not has_login and not has_api_key:
            raise HTTPException(status_code=400, detail={"error": "email+password or api_key is required"})
        server = sub2api_config.add_server(
            name=body.name,
            base_url=body.base_url,
            email=body.email,
            password=body.password,
            api_key=body.api_key,
            group_id=body.group_id,
        )
        return {"server": sanitize_sub2api_server(server), "servers": sanitize_sub2api_servers(sub2api_config.list_servers())}

    @router.post("/api/sub2api/servers/{server_id}")
    async def update_sub2api_server(server_id: str, body: Sub2APIServerUpdateRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        server = sub2api_config.update_server(server_id, body.model_dump(exclude_none=True))
        if server is None:
            raise HTTPException(status_code=404, detail={"error": "server not found"})
        return {"server": sanitize_sub2api_server(server), "servers": sanitize_sub2api_servers(sub2api_config.list_servers())}

    @router.delete("/api/sub2api/servers/{server_id}")
    async def delete_sub2api_server(server_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        if not sub2api_config.delete_server(server_id):
            raise HTTPException(status_code=404, detail={"error": "server not found"})
        return {"servers": sanitize_sub2api_servers(sub2api_config.list_servers())}

    @router.get("/api/sub2api/servers/{server_id}/groups")
    async def sub2api_server_groups(server_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        server = sub2api_config.get_server(server_id)
        if server is None:
            raise HTTPException(status_code=404, detail={"error": "server not found"})
        try:
            groups = await run_in_threadpool(sub2api_list_remote_groups, server)
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc
        return {"server_id": server_id, "groups": groups}

    @router.get("/api/sub2api/servers/{server_id}/accounts")
    async def sub2api_server_accounts(server_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        server = sub2api_config.get_server(server_id)
        if server is None:
            raise HTTPException(status_code=404, detail={"error": "server not found"})
        try:
            accounts = await run_in_threadpool(sub2api_list_remote_accounts, server)
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc
        return {"server_id": server_id, "accounts": accounts}

    @router.post("/api/sub2api/servers/{server_id}/import")
    async def sub2api_server_import(server_id: str, body: Sub2APIImportRequest, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        server = sub2api_config.get_server(server_id)
        if server is None:
            raise HTTPException(status_code=404, detail={"error": "server not found"})
        try:
            job = sub2api_import_service.start_import(server, body.account_ids)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        return {"import_job": job}

    @router.get("/api/sub2api/servers/{server_id}/import")
    async def sub2api_server_import_progress(server_id: str, authorization: str | None = Header(default=None)):
        require_admin(authorization)
        server = sub2api_config.get_server(server_id)
        if server is None:
            raise HTTPException(status_code=404, detail={"error": "server not found"})
        return {"import_job": server.get("import_job")}

    return router
