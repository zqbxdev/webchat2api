from __future__ import annotations

import sys
import types
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch
from typing import Any

if "sqlalchemy" not in sys.modules:
    sqlalchemy = types.ModuleType("sqlalchemy")
    sqlalchemy.Column = lambda *args, **kwargs: None
    sqlalchemy.String = lambda *args, **kwargs: str
    sqlalchemy.Text = lambda *args, **kwargs: str
    sqlalchemy.Integer = lambda *args, **kwargs: int
    sqlalchemy.create_engine = lambda *args, **kwargs: None
    sqlalchemy.text = lambda value: value
    sqlalchemy_ext = types.ModuleType("sqlalchemy.ext")
    sqlalchemy_declarative = types.ModuleType("sqlalchemy.ext.declarative")

    class BaseStub:
        metadata = types.SimpleNamespace(create_all=lambda *args, **kwargs: None)

    sqlalchemy_declarative.declarative_base = lambda: BaseStub
    sqlalchemy_orm = types.ModuleType("sqlalchemy.orm")
    sqlalchemy_orm.sessionmaker = lambda *args, **kwargs: None
    sys.modules["sqlalchemy"] = sqlalchemy
    sys.modules["sqlalchemy.ext"] = sqlalchemy_ext
    sys.modules["sqlalchemy.ext.declarative"] = sqlalchemy_declarative
    sys.modules["sqlalchemy.orm"] = sqlalchemy_orm

if "curl_cffi" not in sys.modules:
    curl_cffi = types.ModuleType("curl_cffi")
    requests_module = types.SimpleNamespace(
        Session=object,
        Response=object,
        exceptions=types.SimpleNamespace(RequestException=Exception),
    )
    curl_cffi.requests = requests_module
    sys.modules["curl_cffi"] = curl_cffi
    sys.modules["curl_cffi.requests"] = requests_module

if "git" not in sys.modules:
    git = types.ModuleType("git")
    git.Repo = object
    git_exc = types.ModuleType("git.exc")
    git_exc.GitCommandError = Exception
    sys.modules["git"] = git
    sys.modules["git.exc"] = git_exc

if "tiktoken" not in sys.modules:
    tiktoken = types.ModuleType("tiktoken")
    tiktoken.encoding_for_model = lambda *args, **kwargs: types.SimpleNamespace(encode=lambda text: [])
    tiktoken.get_encoding = lambda *args, **kwargs: types.SimpleNamespace(encode=lambda text: [])
    sys.modules["tiktoken"] = tiktoken

if "fastapi" not in sys.modules:
    fastapi = types.ModuleType("fastapi")
    from test.optional_stubs import StubHTTPException as HTTPException

    fastapi.HTTPException = HTTPException
    sys.modules["fastapi"] = fastapi

if "fastapi.concurrency" not in sys.modules:
    fastapi_concurrency = types.ModuleType("fastapi.concurrency")
    fastapi_concurrency.run_in_threadpool = lambda func, *args, **kwargs: func(*args, **kwargs)
    sys.modules["fastapi.concurrency"] = fastapi_concurrency

if "fastapi.responses" not in sys.modules:
    fastapi_responses = types.ModuleType("fastapi.responses")
    fastapi_responses.JSONResponse = object
    fastapi_responses.StreamingResponse = object
    sys.modules["fastapi.responses"] = fastapi_responses


class TestGrokConsoleError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        upstream_status: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.upstream_status = upstream_status
        self.code = code

    @property
    def is_check_unavailable(self) -> bool:
        return self.code in {
            "cloudflare_challenge",
            "invalid_rate_limit_response",
            "rate_limit_check_unavailable",
            "rate_limit_network_error",
            "upstream_transient",
        }


_MISSING = object()


@contextmanager
def patched_grok_validation(return_value: object = None, side_effect: Exception | None = None):
    module_name = "services.providers.grok"
    grok_module = types.ModuleType(module_name)
    validate = Mock(return_value=return_value, side_effect=side_effect)
    setattr(grok_module, "GrokConsoleError", TestGrokConsoleError)
    setattr(grok_module, "validate_grok_access_token", validate)
    previous_module = sys.modules.get(module_name, _MISSING)
    services_providers = sys.modules.get("services.providers")
    previous_attr = getattr(services_providers, "grok", _MISSING) if services_providers is not None else _MISSING
    sys.modules[module_name] = grok_module
    if services_providers is not None:
        setattr(services_providers, "grok", grok_module)
    try:
        yield validate
    finally:
        if previous_module is _MISSING:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        if services_providers is not None:
            if previous_attr is _MISSING:
                try:
                    delattr(services_providers, "grok")
                except AttributeError:
                    pass
            else:
                setattr(services_providers, "grok", previous_attr)


@contextmanager
def patched_grok_app_chat_validation(events: list[dict[str, Any]] | None = None, side_effect: Exception | None = None):
    validation = Mock(return_value=iter(events if events is not None else [{"result": {"response": {"token": "pong"}}}]), side_effect=side_effect)
    with patch.object(account_service_module, "_run_grok_app_chat_validation", validation):
        yield validation


from services.account_service import AccountService
import services.account_service as account_service_module
from services.providers import registry as provider_registry
from services.providers.base import AccountStateDecisionInput, decide_account_state
from services.models import GEMINI_PROVIDER, GROK_PROVIDER, GPT_PROVIDER, resolve_model

account_service_module.log_service.add = lambda *args, **kwargs: None


class MemoryStorage:
    def __init__(self, accounts: list[dict[str, Any]] | None = None) -> None:
        self.accounts = list(accounts or [])

    def load_accounts(self) -> list[dict[str, Any]]:
        return list(self.accounts)

    def save_accounts(self, accounts: list[dict[str, Any]]) -> None:
        self.accounts = list(accounts)

    def load_auth_keys(self) -> list[dict[str, Any]]:
        return []

    def save_auth_keys(self, auth_keys: list[dict[str, Any]]) -> None:
        pass

    def health_check(self) -> dict[str, Any]:
        return {"ok": True}

    def get_backend_info(self) -> dict[str, Any]:
        return {"type": "memory"}


class AccountProviderTests(unittest.TestCase):
    def test_normalizes_provider_without_replacing_plan_type(self) -> None:
        service = AccountService(MemoryStorage([{"access_token": "token-1", "type": "plus"}]))

        [account] = service.list_accounts()

        self.assertEqual(account["provider"], GPT_PROVIDER)
        self.assertEqual(account["type"], "plus")

    def test_selects_text_token_by_provider(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "gpt-token", "type": "plus", "provider": "gpt"},
            {"access_token": "sso=grok-token", "type": "basic", "provider": "grok"},
        ])

        self.assertEqual(service.get_text_access_token(provider=GPT_PROVIDER), "gpt-token")
        self.assertEqual(service.get_text_access_token(provider=GROK_PROVIDER), "grok-token")

    def test_selects_gemini_text_token_by_provider(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "gpt-token", "provider": "gpt"},
            {"__Secure-1PSID": "psid", "__Secure-1PSIDTS": "psidts", "provider": "gemini"},
        ])

        self.assertEqual(service.get_text_access_token(provider=GEMINI_PROVIDER), "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts")
        [gemini_account] = [item for item in service.list_accounts() if item["provider"] == GEMINI_PROVIDER]
        self.assertEqual(gemini_account["cookies"], {"__Secure-1PSID": "psid", "__Secure-1PSIDTS": "psidts"})

    def test_gemini_provider_alias_normalizes_to_gemini(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts", "provider": "google"},
        ])

        [account] = service.list_accounts()
        self.assertEqual(account["provider"], GEMINI_PROVIDER)
        self.assertEqual(service.get_text_access_token(provider="gemini"), "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts")

    def test_unknown_account_provider_is_rejected(self) -> None:
        service = AccountService(MemoryStorage())
        result = service.add_account_items([
            {"access_token": "mystery-token", "provider": "mystery"},
        ])

        self.assertEqual(result["added"], 0)
        self.assertEqual(service.list_accounts(), [])

    def test_unknown_requested_text_provider_does_not_select_gpt_token(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "gpt-token", "provider": "gpt"},
        ])

        with self.assertRaises(ValueError):
            service.get_text_access_token(provider="mystery")

    def test_gemini_provider_delete_accepts_cookie_field_token_without_set_mutation(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"__Secure-1PSID": "psid", "__Secure-1PSIDTS": "psidts", "provider": "gemini"},
        ])

        try:
            result = service.delete_accounts(["__Secure-1PSID=psid"], provider=GEMINI_PROVIDER)
        except RuntimeError as exc:
            self.fail(f"Gemini delete mutated target token set while iterating: {exc}")

        self.assertEqual(result["removed"], 1)
        self.assertEqual(service.list_accounts(provider=GEMINI_PROVIDER), [])

    def test_gemini_provider_delete_accepts_normalized_access_token(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"__Secure-1PSID": "psid", "__Secure-1PSIDTS": "psidts", "provider": "gemini"},
        ])

        result = service.delete_accounts(["__Secure-1PSID=psid; __Secure-1PSIDTS=psidts"], provider=GEMINI_PROVIDER)

        self.assertEqual(result["removed"], 1)
        self.assertEqual(service.list_accounts(provider=GEMINI_PROVIDER), [])

    def test_gemini_provider_delete_accepts_full_cookie_header(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"__Secure-1PSID": "psid", "__Secure-1PSIDTS": "psidts", "provider": "gemini"},
        ])

        result = service.delete_accounts([
            "SID=ignored; __Secure-1PSID=psid; __Secure-1PSIDTS=psidts; NID=ignored",
        ], provider=GEMINI_PROVIDER)

        self.assertEqual(result["removed"], 1)
        self.assertEqual(service.list_accounts(provider=GEMINI_PROVIDER), [])

    def test_gemini_account_normalizes_cookie_sources_and_metadata(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {
                "access_token": "SID=sid; __Secure-1PSID=psid; NID=nid; SNlM0e=embedded-session",
                "cookies": {"__Secure-1PSIDTS": "psidts", "CONSENT": "yes", "at": "cookie-at"},
                "session_token": "top-session",
                "provider": "gemini",
            },
        ])

        [account] = service.list_accounts(provider=GEMINI_PROVIDER)
        self.assertEqual(account["access_token"], "SID=sid; __Secure-1PSID=psid; NID=nid; SNlM0e=embedded-session")
        self.assertEqual(account["__Secure-1PSID"], "psid")
        self.assertEqual(account["__Secure-1PSIDTS"], "psidts")
        self.assertEqual(account["cookies"]["SID"], "sid")
        self.assertEqual(account["cookies"]["NID"], "nid")
        self.assertEqual(account["cookies"]["CONSENT"], "yes")
        self.assertEqual(account["cookies"]["SNlM0e"], "embedded-session")
        self.assertEqual(account["cookies"]["at"], "cookie-at")
        self.assertEqual(account["session_token"], "top-session")
        self.assertEqual(account["SNlM0e"], "embedded-session")
        self.assertEqual(account["at"], "cookie-at")
        self.assertTrue(account["has_gemini_session"])
        self.assertEqual(account["account_category"], "full_session")
        self.assertEqual(account["account_status"], "usable_gemini_session")

    def test_gemini_account_categories_cover_session_shapes(self) -> None:
        strategy = provider_registry.account_strategy(GEMINI_PROVIDER)

        cases = [
            ({}, "missing_session", "missing_gemini_session", False),
            ({"__Secure-1PSID": "psid"}, "psid_only", "usable_gemini_session", True),
            ({"__Secure-1PSID": "psid", "__Secure-1PSIDTS": "psidts"}, "psid_psidts", "usable_gemini_session", True),
            (
                {"access_token": "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts; SNlM0e=session"},
                "full_session",
                "usable_gemini_session",
                True,
            ),
        ]

        for account, category, status, has_session in cases:
            with self.subTest(category=category):
                normalized = strategy.normalize_account({**account, "provider": GEMINI_PROVIDER, "access_token": strategy.normalize_access_token(account)}) if account else account
                source = normalized or account
                self.assertEqual(strategy.account_category(source), category)
                self.assertEqual(strategy.account_status(source), status)
                self.assertEqual(strategy.has_gemini_session(source), has_session)

    def test_gemini_text_selection_skips_missing_psid_accounts(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "SNlM0e=session-only", "provider": GEMINI_PROVIDER},
            {"__Secure-1PSID": "psid", "__Secure-1PSIDTS": "psidts", "provider": GEMINI_PROVIDER},
        ])

        accounts = service.list_accounts(provider=GEMINI_PROVIDER)
        self.assertEqual(accounts[0]["account_category"], "session_token_only")
        self.assertEqual(accounts[0]["account_status"], "missing_psid")
        self.assertEqual(service.get_text_access_token(provider=GEMINI_PROVIDER), "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts")

    def test_gemini_refresh_by_sanitized_row_id_uses_client_helpers(self) -> None:
        strategy = provider_registry.account_strategy(GEMINI_PROVIDER)
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"__Secure-1PSID": "psid", "__Secure-1PSIDTS": "old-psidts", "provider": GEMINI_PROVIDER, "status": "正常"},
            {"__Secure-1PSID": "other-psid", "__Secure-1PSIDTS": "other-psidts", "provider": GEMINI_PROVIDER, "status": "正常"},
        ])
        accounts = service.list_accounts(provider=GEMINI_PROVIDER)
        row_id = strategy.sanitize_account(accounts[0])["row_id"]
        client = Mock()
        client.cookie_header = "__Secure-1PSID=psid; __Secure-1PSIDTS=new-psidts; NID=nid"
        client.bootstrap_session_token.return_value = "new-at"
        client_context = Mock()
        client_context.__enter__ = Mock(return_value=client)
        client_context.__exit__ = Mock(return_value=None)

        with patch("services.providers.gemini.client.GeminiWebClient", return_value=client_context) as client_class:
            result = service.refresh_accounts([], provider=GEMINI_PROVIDER, identifiers=[{"row_id": row_id}])

        self.assertTrue(strategy.supports_refresh(accounts[0]))
        self.assertEqual(result["refreshed"], 1)
        self.assertEqual(result["errors"], [])
        client_class.assert_called_once_with("__Secure-1PSID=psid; __Secure-1PSIDTS=old-psidts", None, account_proxy="")
        client.rotate_psidts.assert_called_once_with()
        client.bootstrap_session_token.assert_called_once_with()
        refreshed_accounts = service.list_accounts(provider=GEMINI_PROVIDER)
        self.assertEqual(refreshed_accounts[0]["__Secure-1PSIDTS"], "new-psidts")
        self.assertEqual(refreshed_accounts[0]["session_token"], "new-at")
        self.assertEqual(refreshed_accounts[0]["cookies"], {"__Secure-1PSID": "psid", "__Secure-1PSIDTS": "new-psidts", "NID": "nid"})
        self.assertNotIn("session_token", refreshed_accounts[1])

    def test_gemini_delete_by_sanitized_row_id_without_secret_token(self) -> None:
        strategy = provider_registry.account_strategy(GEMINI_PROVIDER)
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"__Secure-1PSID": "psid", "__Secure-1PSIDTS": "psidts", "provider": GEMINI_PROVIDER, "account_id": "gemini-a"},
            {"__Secure-1PSID": "other-psid", "__Secure-1PSIDTS": "other-psidts", "provider": GEMINI_PROVIDER, "account_id": "gemini-b"},
        ])
        [target, _other] = service.list_accounts(provider=GEMINI_PROVIDER)
        sanitized = strategy.sanitize_account(target)

        result = service.delete_accounts([], provider=GEMINI_PROVIDER, identifiers=[{"row_id": sanitized["row_id"]}])

        self.assertEqual(result["removed"], 1)
        remaining = service.list_accounts(provider=GEMINI_PROVIDER)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["account_id"], "gemini-b")

    def test_gemini_auth_failure_detection_and_refresh_error_message(self) -> None:
        strategy = provider_registry.account_strategy(GEMINI_PROVIDER)

        self.assertTrue(strategy.is_auth_failure_payload({"status_code": 401, "detail": "Unauthorized"}))
        self.assertTrue(strategy.is_auth_failure_payload({"error": {"message": "SNlM0e not found"}}))
        self.assertTrue(strategy.is_auth_failure_payload([{"code": "gemini_auth_failed"}]))
        self.assertFalse(strategy.is_auth_failure_payload({"status_code": 429, "message": "rate limited"}))
        self.assertFalse(strategy.is_auth_failure_payload({"message": "token budget was exceeded"}))
        self.assertEqual(strategy.refresh_error_message(RuntimeError("rotate failed")), "rotate failed")
        self.assertEqual(strategy.refresh_error_message(Exception()), "Gemini session refresh failed")

    def test_gemini_sanitization_redacts_secrets_and_preserves_metadata(self) -> None:
        strategy = provider_registry.account_strategy(GEMINI_PROVIDER)
        account = strategy.normalize_account({
            "accessToken": "SID=sid; __Secure-1PSID=psid; __Secure-1PSIDTS=psidts; SNlM0e=session",
            "cookies": {"NID": "nid", "at": "at-token"},
            "provider": GEMINI_PROVIDER,
            "account_id": "account-a",
        })

        sanitized = strategy.sanitize_account(account)

        for key in ("access_token", "accessToken", "cookies", "__Secure-1PSID", "__Secure-1PSIDTS", "SNlM0e", "session_token", "at"):
            self.assertNotIn(key, sanitized)
        self.assertEqual(sanitized["account_id"], "account-a")
        self.assertTrue(sanitized["row_id"])
        self.assertTrue(sanitized["has_gemini_session"])
        self.assertEqual(sanitized["account_category"], "full_session")
        self.assertEqual(sanitized["account_status"], "usable_gemini_session")

    def test_grok_account_normalizes_simple_sso_cookie_token(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": " sso=grok-token ", "provider": "grok"},
        ])

        [account] = service.list_accounts()
        self.assertEqual(account["access_token"], "grok-token")
        self.assertEqual(service.get_text_access_token(provider=GROK_PROVIDER), "grok-token")

    def test_grok_account_normalizes_case_insensitive_spaced_sso_token(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": " SSO  =  grok-token ", "provider": "grok"},
        ])

        [account] = service.list_accounts()
        self.assertEqual(account["access_token"], "grok-token")
        self.assertEqual(service.get_text_access_token(provider=GROK_PROVIDER), "grok-token")

    def test_grok_account_preserves_spaced_sso_cookie_header_token(self) -> None:
        cookie_token = " SSO  =  grok-token ; other=value "
        service = AccountService(MemoryStorage([{"access_token": cookie_token, "provider": "grok"}]))

        [account] = service.list_accounts()
        self.assertEqual(account["access_token"], "SSO  =  grok-token ; other=value")

    def test_grok_account_preserves_full_cookie_header_token(self) -> None:
        cookie_token = " sso=grok-token ; other=value "
        service = AccountService(MemoryStorage([{"access_token": cookie_token, "provider": "grok"}]))

        [account] = service.list_accounts()
        self.assertEqual(account["access_token"], "sso=grok-token ; other=value")

    def test_grok_sso_import_flag_normalizes_prefixed_token_to_bare_value(self) -> None:
        service = AccountService(MemoryStorage())
        result = service.add_accounts(["sso=grok-simple-cookie-token"], provider=GROK_PROVIDER)

        self.assertEqual(result["added"], 1)
        self.assertEqual(service.list_tokens(provider=GROK_PROVIDER), ["grok-simple-cookie-token"])

    def test_grok_sso_import_flag_accepts_plain_value_without_dot(self) -> None:
        service = AccountService(MemoryStorage())
        result = service.add_accounts(["grok-simple-cookie-token"], provider=GROK_PROVIDER)

        self.assertEqual(result["added"], 1)
        self.assertEqual(service.list_tokens(provider=GROK_PROVIDER), ["grok-simple-cookie-token"])

    def test_grok_account_rejects_plain_token_without_sso_cookie(self) -> None:
        service = AccountService(MemoryStorage())
        result = service.add_account_items([
            {"access_token": "grok-token", "provider": "grok"},
            {"access_token": "sso-rw=grok-token", "provider": "grok"},
        ])

        self.assertEqual(result["added"], 0)
        self.assertEqual(service.list_accounts(provider=GROK_PROVIDER), [])

    def test_grok_account_imports_explicit_sso_payload(self) -> None:
        service = AccountService(MemoryStorage())
        result = service.add_account_items([
            {"sso": "grok-sso-token", "provider": "grok", "tier": "premium"},
            {"sso": "sso=grok-simple-cookie-token", "provider": "grok"},
            {"sso": "sso=grok-cookie-header-token; other=value", "provider": "grok"},
        ])

        self.assertEqual(result["added"], 3)
        accounts = service.list_accounts(provider=GROK_PROVIDER)
        self.assertEqual([account["access_token"] for account in accounts], [
            "grok-sso-token",
            "grok-simple-cookie-token",
            "sso=grok-cookie-header-token; other=value",
        ])
        self.assertEqual(accounts[0]["tier"], "super")

    def test_grok_account_imports_sso_alias_payloads(self) -> None:
        service = AccountService(MemoryStorage())
        result = service.add_account_items([
            {"raw_sso": "raw-sso-token", "provider": "grok"},
            {"rawSso": "raw-camel-sso-token", "provider": "grok"},
            {"sso_token": "sso-token-field", "provider": "grok"},
            {"ssoToken": "sso-camel-token-field", "provider": "grok"},
            {"raw_sso": "sso=raw-cookie-token", "provider": "grok"},
            {"sso_token": "sso=alias-cookie-token; other=value", "provider": "grok"},
        ])

        self.assertEqual(result["added"], 6)
        tokens = service.list_tokens(provider=GROK_PROVIDER)
        self.assertEqual(tokens, [
            "raw-sso-token",
            "raw-camel-sso-token",
            "sso-token-field",
            "sso-camel-token-field",
            "raw-cookie-token",
            "sso=alias-cookie-token; other=value",
        ])

    def test_grok_account_imports_sso_from_token_and_cookies_dict(self) -> None:
        service = AccountService(MemoryStorage())
        result = service.add_account_items([
            {"token": "sso=grok-token-field", "provider": "grok"},
            {"cookies": {"sso": "grok-cookies-dict-token", "sso-rw": "rw-token"}, "provider": "grok"},
            {"cookies": {"SSO": "sso=grok-cookies-dict-cookie-token"}, "provider": "grok"},
        ])

        self.assertEqual(result["added"], 3)
        tokens = service.list_tokens(provider=GROK_PROVIDER)
        self.assertEqual(tokens, [
            "grok-token-field",
            "grok-cookies-dict-token",
            "grok-cookies-dict-cookie-token",
        ])

    def test_grok_account_rejects_explicit_sso_cookie_like_payload_without_sso_pair(self) -> None:
        service = AccountService(MemoryStorage())
        result = service.add_account_items([
            {"sso": "sso-rw=grok-token", "provider": "grok"},
            {"sso": "not-sso=grok-token", "provider": "grok"},
        ])

        self.assertEqual(result["added"], 0)
        self.assertEqual(service.list_accounts(provider=GROK_PROVIDER), [])

    def test_grok_account_imports_cookie_field_with_sso(self) -> None:
        service = AccountService(MemoryStorage())
        result = service.add_account_items([
            {"cookie": "sso=grok-cookie-token; sso-rw=rw-token", "provider": "grok"},
            {"cookies": "sso=grok-cookies-token; other=value", "provider": "grok"},
        ])

        self.assertEqual(result["added"], 2)
        tokens = service.list_tokens(provider=GROK_PROVIDER)
        self.assertEqual(tokens, [
            "sso=grok-cookie-token; sso-rw=rw-token",
            "sso=grok-cookies-token; other=value",
        ])

    def test_grok_account_rejects_cookie_field_without_sso(self) -> None:
        service = AccountService(MemoryStorage())
        result = service.add_account_items([
            {"cookie": "sso-rw=grok-token", "provider": "grok"},
            {"token": "grok-token", "provider": "grok"},
            {"cookies": {"sso-rw": "rw-token"}, "provider": "grok"},
            {"cookies": {"not-sso": "grok-token"}, "provider": "grok"},
        ])

        self.assertEqual(result["added"], 0)
        self.assertEqual(service.list_accounts(provider=GROK_PROVIDER), [])

    def test_grok_account_cloudflare_fields_are_optional_metadata(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok"},
        ])

        [account] = service.list_accounts()
        self.assertEqual(account["access_token"], "grok-token")
        self.assertEqual(account["cf_cookies"], "")
        self.assertIsNone(account["user_agent"])
        self.assertIsNone(account["last_check_status"])
        self.assertIsNone(account["expired_at"])
        self.assertEqual(service.get_text_access_token(provider=GROK_PROVIDER), "grok-token")

    def test_grok_console_quota_normalizes_defaults_and_persists(self) -> None:
        storage = MemoryStorage([{"access_token": "sso=grok-token", "provider": "grok"}])
        service = AccountService(storage)

        [account] = service.list_accounts()
        self.assertEqual(account["quota_console"], {
            "remaining": 30,
            "total": 30,
            "window_seconds": 900,
            "reset_at": None,
        })
        service.get_grok_console_access_token()
        [stored] = storage.accounts
        self.assertEqual(stored["quota_console"]["remaining"], 29)
        self.assertEqual(stored["quota"], 0)

    def test_grok_console_selection_skips_exhausted_until_reset(self) -> None:
        now = [1000.0]
        service = AccountService(MemoryStorage([
            {
                "access_token": "sso=exhausted-token",
                "provider": "grok",
                "quota_console": {"remaining": 0, "total": 30, "window_seconds": 900, "reset_at": 1900},
            },
            {"access_token": "sso=ready-token", "provider": "grok"},
        ]), now=lambda: now[0])

        self.assertEqual(service.get_grok_console_access_token(), "ready-token")
        now[0] = 1900.0
        self.assertEqual(service.get_grok_console_access_token(excluded_tokens={"ready-token"}), "exhausted-token")
        [reset_account] = [item for item in service.list_accounts() if item["access_token"] == "exhausted-token"]
        self.assertEqual(reset_account["quota_console"]["remaining"], 29)
        self.assertIsNone(reset_account["quota_console"]["reset_at"])

    def test_grok_console_selection_reserves_quota_and_sets_reset_window(self) -> None:
        now = [5000.0]
        service = AccountService(MemoryStorage([
            {
                "access_token": "sso=grok-token",
                "provider": "grok",
                "quota_console": {"remaining": 1, "total": 30, "window_seconds": 900, "reset_at": None},
            }
        ]), now=lambda: now[0])

        self.assertEqual(service.get_grok_console_access_token(), "grok-token")

        [account] = service.list_accounts()
        self.assertEqual(account["quota_console"]["remaining"], 0)
        self.assertEqual(account["quota_console"]["reset_at"], 5900)
        self.assertEqual(service.get_grok_console_access_token(), "")

    def test_grok_console_selection_reserves_without_provider_mark(self) -> None:
        service = AccountService(MemoryStorage([
            {
                "access_token": "sso=grok-token",
                "provider": "grok",
                "quota_console": {"remaining": 1, "total": 1, "window_seconds": 900, "reset_at": None},
            }
        ]))

        self.assertEqual(service.get_grok_console_access_token(), "grok-token")
        self.assertEqual(service.get_grok_console_access_token(), "")

        [account] = service.list_accounts()
        self.assertEqual(account["quota_console"]["remaining"], 0)

    def test_text_and_image_usage_do_not_consume_console_quota(self) -> None:
        storage = MemoryStorage([
            {"access_token": "sso=grok-token", "provider": "grok"},
            {"access_token": "gpt-token", "provider": "gpt", "quota": 1},
        ])
        service = AccountService(storage)

        service.mark_text_used("grok-token")
        service.mark_image_result("gpt-token", success=True)

        grok_account = service.get_account("grok-token")
        gpt_account = service.get_account("gpt-token")
        self.assertEqual(grok_account["quota_console"]["remaining"], 30)
        self.assertEqual(grok_account["status"], "正常")
        self.assertTrue(grok_account["app_chat"])
        self.assertEqual(grok_account["last_check_status"], "valid_by_call")
        self.assertIsNotNone(grok_account["last_success_at"])
        self.assertNotIn("quota_console", gpt_account)
        self.assertEqual(gpt_account["quota"], 0)

    def test_grok_sanitization_preserves_lifecycle_metadata(self) -> None:
        sanitized = provider_registry.account_strategy(GROK_PROVIDER).sanitize_account({
            "access_token": "sso=secret",
            "sso": "secret",
            "cf_clearance": "cf-secret",
            "state_reason": "cloudflare_or_forbidden",
            "last_check_status": "unverified",
            "last_check_http_status": 403,
        })
        self.assertNotIn("access_token", sanitized)
        self.assertNotIn("sso", sanitized)
        self.assertEqual(sanitized["state_reason"], "cloudflare_or_forbidden")
        self.assertEqual(sanitized["last_check_status"], "unverified")
        self.assertEqual(sanitized["last_check_http_status"], 403)

    def test_mark_text_used_keeps_disabled_grok_disabled(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "禁用"},
        ])

        service.mark_text_used("grok-token")

        account = service.get_account("grok-token", provider=GROK_PROVIDER)
        self.assertEqual(account["status"], "禁用")
        self.assertTrue(account["app_chat"])
        self.assertEqual(account["last_check_status"], "valid_by_call")

    def test_grok_console_usage_updates_lifecycle_health(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {
                "access_token": "sso=grok-token",
                "provider": "grok",
                "status": "正常",
                "state_reason": "cloudflare_or_forbidden",
                "last_check_status": "unverified",
                "last_check_error": "rate_limit_check_unavailable",
                "refresh_backoff_until": "2099-01-01 00:00:00",
            },
        ])

        service.mark_grok_console_used("grok-token", success=True)

        account = service.get_account("grok-token", provider=GROK_PROVIDER)
        self.assertEqual(account["status"], "正常")
        self.assertEqual(account["success"], 1)
        self.assertEqual(account["last_check_status"], "valid_by_call")
        self.assertEqual(account["last_check_error"], "")
        self.assertEqual(account["state_reason"], "")
        self.assertIsNone(account["refresh_backoff_until"])
        self.assertIsNotNone(account["last_success_at"])

    def test_grok_console_usage_keeps_disabled_grok_disabled(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "禁用"},
        ])

        service.mark_grok_console_used("grok-token", success=True)

        account = service.get_account("grok-token", provider=GROK_PROVIDER)
        self.assertEqual(account["status"], "禁用")
        self.assertEqual(account["last_check_status"], "valid_by_call")
        self.assertIsNotNone(account["last_success_at"])

    def test_selects_grok_app_chat_token_by_tier_semantics(self) -> None:
        basic_service = AccountService(MemoryStorage())
        basic_service.add_account_items([
            {"access_token": "sso=basic-token", "provider": "grok", "tier": "free"},
            {"access_token": "sso=super-token", "provider": "grok", "tier": "premium"},
            {"access_token": "sso=heavy-token", "provider": "grok", "tier": "heavy"},
        ])
        super_service = AccountService(MemoryStorage())
        super_service.add_account_items([
            {"access_token": "sso=basic-token", "provider": "grok", "tier": "free"},
            {"access_token": "sso=super-token", "provider": "grok", "tier": "premium"},
            {"access_token": "sso=heavy-token", "provider": "grok", "tier": "heavy"},
        ])
        heavy_service = AccountService(MemoryStorage())
        heavy_service.add_account_items([
            {"access_token": "sso=basic-token", "provider": "grok", "tier": "free"},
            {"access_token": "sso=super-token", "provider": "grok", "tier": "premium"},
            {"access_token": "sso=heavy-token", "provider": "grok", "tier": "heavy"},
        ])

        self.assertEqual(basic_service.get_grok_app_chat_access_token(resolve_model("grok-4.20-0309-non-reasoning")), "basic-token")
        self.assertEqual(super_service.get_grok_app_chat_access_token(resolve_model("grok-4.20-0309")), "super-token")
        self.assertEqual(heavy_service.get_grok_app_chat_access_token(resolve_model("grok-4.20-0309-heavy")), "heavy-token")

    def test_prefer_best_grok_app_chat_selection_tries_heavy_super_basic(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=basic-token", "provider": "grok", "tier": "basic", "cf_cookies": "cf_bm=value", "user_agent": "Test UA"},
            {"access_token": "sso=super-token", "provider": "grok", "tier": "super"},
        ])

        [account] = [item for item in service.list_accounts() if item["access_token"] == "basic-token"]
        self.assertEqual(account["cf_cookies"], "cf_bm=value")
        self.assertEqual(account["user_agent"], "Test UA")
        self.assertEqual(service.get_grok_app_chat_access_token(resolve_model("grok-4.20-fast")), "super-token")

    def test_grok_app_chat_selection_uses_capabilities_and_status(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=disabled-heavy", "provider": "grok", "tier": "heavy", "status": "禁用"},
            {"access_token": "sso=image-heavy", "provider": "grok", "tier": "heavy", "capabilities": ["image"]},
            {"access_token": "sso=chat-heavy", "provider": "grok", "tier": "heavy", "capabilities": "chat,heavy"},
        ])

        self.assertEqual(service.get_grok_app_chat_access_token(resolve_model("grok-4.20-heavy")), "chat-heavy")

    def test_grok_app_chat_selection_falls_back_to_provider_round_robin(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "gpt-token", "provider": "gpt"},
            {"access_token": "sso=unknown-tier", "provider": "grok", "tier": "enterprise"},
            {"access_token": "sso=plain-grok", "provider": "grok"},
        ])

        self.assertEqual(service.get_grok_app_chat_access_token(resolve_model("grok-4.20-0309-heavy")), "unknown-tier")
        self.assertEqual(service.get_grok_app_chat_access_token(resolve_model("grok-4.20-0309-heavy")), "plain-grok")

    def test_grok_app_chat_selection_skips_capability_and_tier_mismatches(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=image-heavy", "provider": "grok", "tier": "heavy", "capabilities": ["image"]},
            {"access_token": "sso=basic-chat", "provider": "grok", "tier": "basic", "capabilities": ["chat"]},
            {"access_token": "sso=heavy-chat", "provider": "grok", "tier": "heavy", "capabilities": ["chat"]},
        ])

        self.assertEqual(service.get_grok_app_chat_access_token(resolve_model("grok-4.20-heavy")), "heavy-chat")

    def test_shared_account_state_decision_covers_auth_rate_limit_and_restore(self) -> None:
        strategy = provider_registry.account_strategy(GROK_PROVIDER)
        account = strategy.normalize_account({
            "access_token": "grok-token",
            "provider": GROK_PROVIDER,
            "status": "限流",
            "quota_console": {"remaining": 0, "total": 3, "window_seconds": 900, "reset_at": 10},
        })

        auth_decision = decide_account_state(AccountStateDecisionInput(account=account, adapter=strategy, status_code=401))
        rate_limit_decision = decide_account_state(AccountStateDecisionInput(account=account, adapter=strategy, status_code=429))
        restore_decision = decide_account_state(AccountStateDecisionInput(account=account, adapter=strategy, current_time=11))

        self.assertTrue(auth_decision.auth_failed)
        self.assertEqual(auth_decision.writeback, {"status": "异常", "quota": 0})
        self.assertTrue(rate_limit_decision.rate_limited)
        self.assertEqual(rate_limit_decision.writeback, {"status": "限流"})
        self.assertEqual(restore_decision.writeback["quota_console"], {"remaining": 3, "total": 3, "window_seconds": 900, "reset_at": None})

    def test_shared_account_state_decision_reports_capability_and_tier_mismatch(self) -> None:
        strategy = provider_registry.account_strategy(GROK_PROVIDER)
        spec = resolve_model("grok-4.20-0309-heavy")

        image_decision = decide_account_state(AccountStateDecisionInput(
            account={"provider": GROK_PROVIDER, "status": "正常", "tier": "heavy", "capabilities": ["image"]},
            adapter=strategy,
            spec=spec,
        ))
        basic_decision = decide_account_state(AccountStateDecisionInput(
            account={"provider": GROK_PROVIDER, "status": "正常", "tier": "basic", "capabilities": ["chat"]},
            adapter=strategy,
            spec=spec,
        ))
        heavy_decision = decide_account_state(AccountStateDecisionInput(
            account={"provider": GROK_PROVIDER, "status": "正常", "tier": "heavy", "capabilities": ["chat"]},
            adapter=strategy,
            spec=spec,
        ))

        self.assertTrue(image_decision.capability_mismatch)
        self.assertTrue(image_decision.skip_for_selection)
        self.assertTrue(basic_decision.tier_mismatch)
        self.assertFalse(basic_decision.matches_tier("heavy"))
        self.assertFalse(heavy_decision.tier_mismatch)
        self.assertTrue(heavy_decision.matches_tier("heavy"))

    def test_refresh_accounts_attempts_grok_validation(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "异常", "quota": 5},
        ])

        with patched_grok_validation(return_value={"rateLimits": []}) as validate:
            refresh_result = service.refresh_accounts(["grok-token"])

        validate.assert_called_once()
        self.assertEqual(validate.call_args.args[0], "grok-token")
        self.assertEqual(refresh_result["refreshed"], 0)
        self.assertEqual(refresh_result["errors"], [])
        [account] = service.list_accounts()
        self.assertEqual(account["status"], "异常")
        self.assertEqual(account["quota"], 5)
        self.assertFalse(account["app_chat"])

    def test_refresh_accounts_counts_grok_validation_with_usable_quota(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "异常"},
        ])

        payload = {"remainingQueries": 17, "totalQueries": 20, "windowSizeSeconds": 7200}
        with patched_grok_validation(return_value=payload) as validate:
            refresh_result = service.refresh_accounts(["grok-token"])

        validate.assert_called_once()
        self.assertEqual(validate.call_args.args[0], "grok-token")
        self.assertEqual(refresh_result["refreshed"], 1)
        self.assertEqual(refresh_result["errors"], [])
        [account] = service.list_accounts()
        self.assertEqual(account["status"], "正常")
        self.assertTrue(account["app_chat"])
        self.assertEqual(account["quota"], 17)
        self.assertEqual(account["quota_console"], {
            "remaining": 17,
            "total": 20,
            "window_seconds": 7200,
            "reset_at": None,
        })
        self.assertEqual(account["rate_limit_window_seconds"], 7200)
        self.assertEqual(account["tier"], "super")

    def test_refresh_accounts_marks_invalid_grok_validation_abnormal(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "正常"},
        ])

        previous_auto_remove = account_service_module.config.data.get("auto_remove_invalid_accounts")
        account_service_module.config.data["auto_remove_invalid_accounts"] = False
        try:
            with patched_grok_validation(side_effect=TestGrokConsoleError("failed to look up session id for sso=secret-cookie", 401, 401, None)):
                refresh_result = service.refresh_accounts(["grok-token"])
        finally:
            if previous_auto_remove is None:
                account_service_module.config.data.pop("auto_remove_invalid_accounts", None)
            else:
                account_service_module.config.data["auto_remove_invalid_accounts"] = previous_auto_remove

        self.assertEqual(refresh_result["refreshed"], 0)
        self.assertEqual(len(refresh_result["errors"]), 1)
        self.assertEqual(refresh_result["errors"][0]["error"], "Grok 账号鉴权失败：SSO/Cookie 可能已失效，已按策略标记为异常")
        self.assertNotIn("secret-cookie", refresh_result["errors"][0]["error"])
        [account] = service.list_accounts()
        self.assertEqual(account["status"], "异常")
        self.assertEqual(account["quota"], 0)

    def test_refresh_accounts_keeps_grok_network_failure_status_unchanged(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "正常", "quota": 9},
        ])

        with patched_grok_validation(side_effect=TestGrokConsoleError("connect failed with sso=secret-cookie", 502, None, "rate_limit_network_error")):
            refresh_result = service.refresh_accounts(["grok-token"])

        self.assertEqual(refresh_result["refreshed"], 0)
        self.assertEqual(refresh_result["errors"], [])
        [account] = service.list_accounts()
        self.assertEqual(account["status"], "正常")
        self.assertEqual(account["quota"], 9)

    def test_refresh_accounts_keeps_grok_invalid_json_status_unchanged(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "正常", "quota": 4},
        ])

        with patched_grok_validation(side_effect=TestGrokConsoleError("<html>Just a moment sso=secret-cookie</html>", 502, None, "invalid_rate_limit_response")):
            refresh_result = service.refresh_accounts(["grok-token"])

        self.assertEqual(refresh_result["refreshed"], 0)
        self.assertEqual(refresh_result["errors"], [])
        [account] = service.list_accounts()
        self.assertEqual(account["status"], "正常")
        self.assertEqual(account["quota"], 4)

    def test_refresh_accounts_keeps_grok_unknown_403_status_unchanged(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "正常", "quota": 6},
        ])

        with patched_grok_validation(side_effect=TestGrokConsoleError("unknown upstream 403 with sso=secret-cookie", 502, 403, "rate_limit_check_unavailable")):
            refresh_result = service.refresh_accounts(["grok-token"])

        self.assertEqual(refresh_result["refreshed"], 0)
        self.assertEqual(refresh_result["checked"], 1)
        self.assertEqual(refresh_result["unchanged"], 1)
        self.assertEqual(refresh_result["failed"], 0)
        self.assertEqual(refresh_result["errors"], [])
        [account] = service.list_accounts()
        self.assertEqual(account["status"], "正常")
        self.assertEqual(account["quota"], 6)
        self.assertEqual(account["last_check_status"], "unverified")
        self.assertEqual(account["state_reason"], "cloudflare_or_forbidden")
        self.assertEqual(account["last_check_error"], "rate_limit_check_unavailable")
        self.assertEqual(account["last_check_http_status"], 403)
        self.assertIsNotNone(account["last_refresh_attempt_at"])
        self.assertIsNotNone(account["refresh_backoff_until"])

    def test_refresh_accounts_preserves_recent_grok_success_on_probe_403(self) -> None:
        now = datetime(2026, 6, 5, 3, 0, 0, tzinfo=timezone.utc)
        service = AccountService(MemoryStorage(), now=lambda: now.timestamp())
        service.add_account_items([
            {
                "access_token": "sso=grok-token",
                "provider": "grok",
                "status": "正常",
                "quota": 6,
                "last_check_status": "valid_by_call",
                "last_success_at": (now - timedelta(minutes=10)).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
            },
        ])

        with patched_grok_validation(side_effect=TestGrokConsoleError("unknown upstream 403 with sso=secret-cookie", 502, 403, "rate_limit_check_unavailable")):
            refresh_result = service.refresh_accounts(["grok-token"])

        self.assertEqual(refresh_result["checked"], 1)
        self.assertEqual(refresh_result["unchanged"], 1)
        self.assertEqual(refresh_result["errors"], [])
        [account] = service.list_accounts()
        self.assertEqual(account["status"], "正常")
        self.assertEqual(account["quota"], 6)
        self.assertEqual(account["last_check_status"], "valid_by_call")
        self.assertEqual(account["state_reason"], "")
        self.assertEqual(account["last_check_error"], "")
        self.assertIsNone(account["last_check_http_status"])
        self.assertEqual(account["last_check_at"], now.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"))
        self.assertIsNotNone(account["last_refresh_attempt_at"])
        self.assertIsNotNone(account["refresh_backoff_until"])

    def test_refresh_accounts_recovers_recent_success_after_prior_unverified_probe(self) -> None:
        now = datetime(2026, 6, 5, 3, 0, 0, tzinfo=timezone.utc)
        service = AccountService(MemoryStorage(), now=lambda: now.timestamp())
        service.add_account_items([
            {
                "access_token": "sso=grok-token",
                "provider": "grok",
                "status": "正常",
                "quota": 6,
                "state_reason": "cloudflare_or_forbidden",
                "last_check_status": "unverified",
                "last_check_error": "cloudflare_challenge",
                "last_check_http_status": 403,
                "last_success_at": (now - timedelta(minutes=10)).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
            },
        ])

        with patched_grok_validation(side_effect=TestGrokConsoleError("unknown upstream 403 with sso=secret-cookie", 502, 403, "rate_limit_check_unavailable")):
            refresh_result = service.refresh_accounts(["grok-token"])

        self.assertEqual(refresh_result["checked"], 1)
        self.assertEqual(refresh_result["unchanged"], 1)
        [account] = service.list_accounts()
        self.assertEqual(account["status"], "正常")
        self.assertEqual(account["last_check_status"], "valid_by_call")
        self.assertEqual(account["state_reason"], "")
        self.assertEqual(account["last_check_error"], "")
        self.assertIsNone(account["last_check_http_status"])
        self.assertEqual(account["last_check_at"], now.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"))
        self.assertIsNotNone(account["refresh_backoff_until"])

    def test_refresh_accounts_allows_unverified_when_grok_success_is_stale(self) -> None:
        now = datetime(2026, 6, 5, 3, 0, 0, tzinfo=timezone.utc)
        service = AccountService(MemoryStorage(), now=lambda: now.timestamp())
        service.add_account_items([
            {
                "access_token": "sso=grok-token",
                "provider": "grok",
                "status": "正常",
                "quota": 6,
                "last_check_status": "valid_by_call",
                "last_success_at": (now - timedelta(days=2)).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
            },
        ])

        with patched_grok_validation(side_effect=TestGrokConsoleError("unknown upstream 403 with sso=secret-cookie", 502, 403, "rate_limit_check_unavailable")):
            refresh_result = service.refresh_accounts(["grok-token"])

        self.assertEqual(refresh_result["checked"], 1)
        self.assertEqual(refresh_result["unchanged"], 1)
        [account] = service.list_accounts()
        self.assertEqual(account["status"], "正常")
        self.assertEqual(account["last_check_status"], "unverified")
        self.assertEqual(account["state_reason"], "cloudflare_or_forbidden")
        self.assertEqual(account["last_check_error"], "rate_limit_check_unavailable")
        self.assertEqual(account["last_check_http_status"], 403)

    def test_refresh_accounts_keeps_grok_unconfirmed_bare_upstream_failures_unchanged(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token-1", "provider": GROK_PROVIDER, "status": "正常", "quota": 6},
            {"access_token": "sso=grok-token-2", "provider": GROK_PROVIDER, "status": "限流", "quota": 3},
        ])

        def validation_failure(access_token: str, _account: dict[str, Any]) -> None:
            if access_token == "grok-token-1":
                raise TestGrokConsoleError("bare upstream 403 with sso=secret-cookie-1", 502, 403, None)
            raise TestGrokConsoleError("unknown validation failed with sso=secret-cookie-2", 502, None, "mystery_failure")

        with patched_grok_validation(side_effect=validation_failure) as validate:
            refresh_result = service.refresh_accounts([], provider=GROK_PROVIDER)

        self.assertEqual(validate.call_count, 2)
        self.assertEqual(refresh_result["checked"], 2)
        self.assertEqual(refresh_result["refreshed"], 0)
        self.assertEqual(refresh_result["unchanged"], 2)
        self.assertEqual(refresh_result["failed"], 0)
        self.assertEqual(refresh_result["errors"], [])
        accounts = service.list_accounts(provider=GROK_PROVIDER)
        self.assertEqual([account["status"] for account in accounts], ["正常", "限流"])
        self.assertEqual([account["quota"] for account in accounts], [6, 3])

    def test_refresh_accounts_marks_grok_confirmed_invalid_marker_abnormal(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "正常", "quota": 5},
        ])

        previous_auto_remove = account_service_module.config.data.get("auto_remove_invalid_accounts")
        account_service_module.config.data["auto_remove_invalid_accounts"] = False
        try:
            with patched_grok_validation(side_effect=TestGrokConsoleError("failed to look up session id for sso=secret-cookie", 401, 401, None)):
                refresh_result = service.refresh_accounts(["grok-token"])
        finally:
            if previous_auto_remove is None:
                account_service_module.config.data.pop("auto_remove_invalid_accounts", None)
            else:
                account_service_module.config.data["auto_remove_invalid_accounts"] = previous_auto_remove

        self.assertEqual(refresh_result["refreshed"], 0)
        self.assertEqual(len(refresh_result["errors"]), 1)
        self.assertEqual(refresh_result["errors"][0]["error"], "Grok 账号鉴权失败：SSO/Cookie 可能已失效，已按策略标记为异常")
        self.assertNotIn("secret-cookie", refresh_result["errors"][0]["error"])
        [account] = service.list_accounts()
        self.assertEqual(account["status"], "异常")
        self.assertEqual(account["quota"], 0)

    def test_refresh_accounts_marks_grok_quota_validation_limited(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "正常"},
        ])

        with patched_grok_validation(side_effect=TestGrokConsoleError("limited with sso=secret-cookie", 429, 429, "rate_limit_exceeded")):
            refresh_result = service.refresh_accounts(["grok-token"])

        self.assertEqual(refresh_result["refreshed"], 0)
        self.assertEqual(len(refresh_result["errors"]), 1)
        self.assertEqual(refresh_result["errors"][0]["error"], "Grok 账号配额不足或触发限流，已按策略标记为限流")
        self.assertNotIn("secret-cookie", refresh_result["errors"][0]["error"])
        [account] = service.list_accounts()
        self.assertEqual(account["status"], "限流")

    def test_refresh_accounts_keeps_valid_grok_validation_usable(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "异常"},
        ])

        with patched_grok_validation(return_value={"rateLimits": [{"remainingQueries": 1, "totalQueries": 1, "windowSizeSeconds": 7200}]}) as validate:
            refresh_result = service.refresh_accounts(["grok-token"])

        validate.assert_called_once()
        self.assertEqual(refresh_result["refreshed"], 1)
        self.assertEqual(refresh_result["checked"], 1)
        self.assertEqual(refresh_result["unchanged"], 0)
        self.assertEqual(refresh_result["failed"], 0)
        self.assertEqual(refresh_result["errors"], [])
        [account] = service.list_accounts()
        self.assertEqual(account["status"], "正常")
        self.assertEqual(account["last_check_status"], "valid")
        self.assertEqual(account["last_check_error"], "")
        self.assertIsNone(account["refresh_backoff_until"])
        self.assertEqual(account["last_refresh_attempt_at"], account["last_refresh_success_at"])
        self.assertEqual(service.get_text_access_token(provider=GROK_PROVIDER), "grok-token")

    def test_validate_accounts_by_grok_row_id_uses_app_chat(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "异常", "quota": 5},
        ])
        [account] = service.list_accounts(provider=GROK_PROVIDER)
        row_id = provider_registry.account_strategy(GROK_PROVIDER).sanitize_account(account)["row_id"]

        with patched_grok_app_chat_validation() as validate:
            result = service.validate_accounts([], provider=GROK_PROVIDER, identifiers=[{"row_id": row_id}])

        validate.assert_called_once()
        self.assertEqual(validate.call_args.args[0], "grok-token")
        self.assertEqual(result["checked"], 1)
        self.assertEqual(result["valid"], 1)
        self.assertEqual(result["invalid"], 0)
        self.assertEqual(result["limited"], 0)
        self.assertEqual(result["unverified"], 0)
        self.assertEqual(result["errors"], [])
        [updated] = service.list_accounts(provider=GROK_PROVIDER)
        self.assertEqual(updated["status"], "正常")
        self.assertTrue(updated["app_chat"])
        self.assertEqual(updated["quota"], 5)
        self.assertEqual(updated["last_check_status"], "valid")
        self.assertIsNotNone(updated["last_success_at"])
        self.assertEqual(updated["last_check_error"], "")

    def test_validate_accounts_marks_invalid_grok_abnormal(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "正常", "quota": 5},
        ])
        previous_auto_remove = account_service_module.config.data.get("auto_remove_invalid_accounts")
        account_service_module.config.data["auto_remove_invalid_accounts"] = False
        try:
            with patched_grok_app_chat_validation(side_effect=TestGrokConsoleError("failed to look up session id for sso=secret-cookie", 401, 401, None)):
                result = service.validate_accounts(["grok-token"], provider=GROK_PROVIDER)
        finally:
            if previous_auto_remove is None:
                account_service_module.config.data.pop("auto_remove_invalid_accounts", None)
            else:
                account_service_module.config.data["auto_remove_invalid_accounts"] = previous_auto_remove

        self.assertEqual(result["checked"], 1)
        self.assertEqual(result["invalid"], 1)
        self.assertNotIn("secret-cookie", str(result))
        [account] = service.list_accounts(provider=GROK_PROVIDER)
        self.assertEqual(account["status"], "异常")
        self.assertEqual(account["quota"], 0)

    def test_validate_accounts_marks_limited_grok_limited(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "正常", "quota": 5},
        ])

        with patched_grok_app_chat_validation(side_effect=TestGrokConsoleError("limited with sso=secret-cookie", 429, 429, "rate_limit_exceeded")):
            result = service.validate_accounts(["grok-token"], provider=GROK_PROVIDER)

        self.assertEqual(result["checked"], 1)
        self.assertEqual(result["limited"], 1)
        self.assertNotIn("secret-cookie", str(result))
        [account] = service.list_accounts(provider=GROK_PROVIDER)
        self.assertEqual(account["status"], "限流")
        self.assertEqual(account["quota"], 5)
        self.assertEqual(account["state_reason"], "rate_limited")
        self.assertEqual(account["last_check_status"], "limited")
        self.assertEqual(account["last_check_error"], "rate_limit_exceeded")
        self.assertEqual(account["last_check_http_status"], 429)
        self.assertIsNotNone(account["cooldown_until"])

    def test_validate_accounts_keeps_transient_grok_unverified(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "正常", "quota": 5},
        ])

        with patched_grok_app_chat_validation(side_effect=TestGrokConsoleError("cloudflare blocked sso=secret-cookie", 502, 403, "cloudflare_challenge")):
            result = service.validate_accounts(["grok-token"], provider=GROK_PROVIDER)

        self.assertEqual(result["checked"], 1)
        self.assertEqual(result["unverified"], 1)
        self.assertNotIn("secret-cookie", str(result))
        [account] = service.list_accounts(provider=GROK_PROVIDER)
        self.assertEqual(account["status"], "正常")
        self.assertEqual(account["quota"], 5)
        self.assertEqual(account["state_reason"], "cloudflare_or_forbidden")
        self.assertEqual(account["last_check_status"], "unverified")
        self.assertEqual(account["last_check_error"], "cloudflare_challenge")
        self.assertEqual(account["last_check_http_status"], 403)
        self.assertIsNotNone(account["refresh_backoff_until"])

    def test_validate_accounts_preserves_recent_grok_success_on_probe_403(self) -> None:
        now = datetime(2026, 6, 5, 3, 0, 0, tzinfo=timezone.utc)
        service = AccountService(MemoryStorage(), now=lambda: now.timestamp())
        service.add_account_items([
            {
                "access_token": "sso=grok-token",
                "provider": "grok",
                "status": "正常",
                "quota": 5,
                "last_check_status": "valid_by_call",
                "last_success_at": (now - timedelta(minutes=10)).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"),
            },
        ])

        with patched_grok_app_chat_validation(side_effect=TestGrokConsoleError("cloudflare blocked sso=secret-cookie", 502, 403, "cloudflare_challenge")):
            result = service.validate_accounts(["grok-token"], provider=GROK_PROVIDER)

        self.assertEqual(result["checked"], 1)
        self.assertEqual(result["unverified"], 1)
        self.assertNotIn("secret-cookie", str(result))
        [account] = service.list_accounts(provider=GROK_PROVIDER)
        self.assertEqual(account["status"], "正常")
        self.assertEqual(account["quota"], 5)
        self.assertEqual(account["last_check_status"], "valid_by_call")
        self.assertEqual(account["state_reason"], "")
        self.assertEqual(account["last_check_error"], "")
        self.assertIsNone(account["last_check_http_status"])
        self.assertEqual(account["last_check_at"], now.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S"))
        self.assertIsNotNone(account["refresh_backoff_until"])

    def test_validate_accounts_rejects_non_grok_provider(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "gpt-token", "provider": GPT_PROVIDER, "status": "正常"},
        ])

        with self.assertRaises(ValueError):
            service.validate_accounts(["gpt-token"], provider=GPT_PROVIDER)

    def test_refresh_accounts_by_grok_row_id_attempts_validation(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "status": "正常"},
        ])
        [account] = service.list_accounts(provider=GROK_PROVIDER)
        row_id = provider_registry.account_strategy(GROK_PROVIDER).sanitize_account(account)["row_id"]

        with patched_grok_validation(return_value={"remainingQueries": 1, "windowSizeSeconds": 7200}) as validate:
            refresh_result = service.refresh_accounts([], provider=GROK_PROVIDER, identifiers=[{"row_id": row_id}])

        validate.assert_called_once()
        self.assertEqual(validate.call_args.args[0], "grok-token")
        self.assertEqual(refresh_result["refreshed"], 1)
        self.assertEqual(refresh_result["errors"], [])

    def test_same_access_token_is_scoped_by_provider(self) -> None:
        storage = MemoryStorage()
        service = AccountService(storage)
        service.add_account_items([
            {"access_token": "shared-token", "provider": GPT_PROVIDER, "status": "正常"},
            {"access_token": "sso=shared-token", "provider": GROK_PROVIDER, "status": "正常"},
            {"access_token": "__Secure-1PSID=shared-token; __Secure-1PSIDTS=ts", "provider": GEMINI_PROVIDER, "status": "正常"},
        ])

        self.assertEqual(len(service.list_accounts()), 3)
        self.assertEqual(len(storage.accounts), 3)
        self.assertEqual(service.list_tokens(provider=GPT_PROVIDER), ["shared-token"])
        self.assertEqual(service.list_tokens(provider=GROK_PROVIDER), ["shared-token"])
        self.assertEqual(service.list_tokens(provider=GEMINI_PROVIDER), ["__Secure-1PSID=shared-token; __Secure-1PSIDTS=ts"])

    def test_add_accounts_can_target_non_default_provider(self) -> None:
        service = AccountService(MemoryStorage())

        result = service.add_accounts(["sso=shared-token"], provider=GROK_PROVIDER)

        self.assertEqual(result["added"], 1)
        self.assertEqual(service.list_tokens(provider=GPT_PROVIDER), [])
        self.assertEqual(service.list_tokens(provider=GROK_PROVIDER), ["shared-token"])
        self.assertEqual(service.get_account("shared-token", provider=GROK_PROVIDER)["provider"], GROK_PROVIDER)
        [account] = service.list_accounts(provider=GROK_PROVIDER)
        self.assertTrue(provider_registry.account_strategy(GROK_PROVIDER).sanitize_account(account)["has_sso"])

    def test_add_accounts_accepts_plain_grok_sso_value(self) -> None:
        service = AccountService(MemoryStorage())

        result = service.add_accounts(["shared-token"], provider=GROK_PROVIDER)

        self.assertEqual(result["added"], 1)
        self.assertEqual(service.list_tokens(provider=GROK_PROVIDER), ["shared-token"])

    def test_structured_grok_raw_sso_accepts_bare_token(self) -> None:
        service = AccountService(MemoryStorage())

        result = service.add_account_items([{"raw_sso": "explicit-bare-sso-token", "provider": GROK_PROVIDER}])

        self.assertEqual(result["added"], 1)
        self.assertEqual(service.list_tokens(provider=GROK_PROVIDER), ["explicit-bare-sso-token"])
        [account] = service.list_accounts(provider=GROK_PROVIDER)
        self.assertEqual(account["access_token"], "explicit-bare-sso-token")
        self.assertTrue(provider_registry.account_strategy(GROK_PROVIDER).sanitize_account(account)["has_sso"])

    def test_provider_scoped_delete_update_refresh_and_export_do_not_cross_providers(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "shared-token", "provider": GPT_PROVIDER, "status": "正常", "quota": 1},
            {"access_token": "sso=shared-token", "provider": GROK_PROVIDER, "status": "正常"},
        ])

        update_result = service.update_account("shared-token", {"status": "禁用"}, provider=GROK_PROVIDER)
        self.assertIsNotNone(update_result)
        self.assertEqual(service.get_account("shared-token", provider=GPT_PROVIDER)["status"], "正常")
        self.assertEqual(service.get_account("shared-token", provider=GROK_PROVIDER)["status"], "禁用")

        with patched_grok_validation(return_value={"rateLimits": [{"remainingQueries": 1, "totalQueries": 1, "windowSizeSeconds": 7200}]}) as validate:
            refresh_result = service.refresh_accounts(["shared-token"], provider=GROK_PROVIDER)
        validate.assert_called_once()
        self.assertEqual(refresh_result["refreshed"], 1)
        self.assertEqual(refresh_result["items"][0]["provider"], GROK_PROVIDER)
        self.assertEqual(service.get_account("shared-token", provider=GPT_PROVIDER)["status"], "正常")

        self.assertEqual(
            [item["access_token"] for item in service.build_export_items(["shared-token"], provider=GROK_PROVIDER)],
            ["shared-token"],
        )
        delete_result = service.delete_accounts(["shared-token"], provider=GROK_PROVIDER)
        self.assertEqual(delete_result["removed"], 1)
        self.assertIsNotNone(service.get_account("shared-token", provider=GPT_PROVIDER))
        self.assertIsNone(service.get_account("shared-token", provider=GROK_PROVIDER))

    def test_delete_limited_accounts_can_filter_by_provider(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "gpt-token", "provider": GPT_PROVIDER, "status": "限流"},
            {"access_token": "sso=grok-token", "provider": GROK_PROVIDER, "status": "限流"},
        ])

        result = service.delete_limited_accounts(provider=GROK_PROVIDER)

        self.assertEqual(result["removed"], 1)
        self.assertEqual([item["access_token"] for item in service.list_accounts()], ["gpt-token"])

    def test_image_and_refresh_candidates_do_not_use_grok_for_image(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-token", "provider": "grok", "quota": 5, "status": "正常", "tier": "supergrok"},
            {"access_token": "gemini-token", "provider": "gemini", "quota": 5, "status": "正常"},
            {"access_token": "gpt-token", "provider": "gpt", "quota": 5, "status": "正常"},
        ])

        self.assertEqual(service._list_ready_candidate_tokens(), ["gpt-token"])
        self.assertEqual(service.list_limited_tokens(), [])

    def test_registry_exposes_independent_account_strategies(self) -> None:
        gpt_strategy = provider_registry.account_strategy(GPT_PROVIDER)
        grok_strategy = provider_registry.account_strategy(GROK_PROVIDER)
        gemini_strategy = provider_registry.account_strategy(GEMINI_PROVIDER)

        self.assertEqual(gpt_strategy.normalize_access_token({"accessToken": "gpt-token"}), "gpt-token")
        self.assertEqual(grok_strategy.normalize_access_token({"access_token": "sso=grok-token"}), "grok-token")
        self.assertEqual(
            gemini_strategy.normalize_access_token({"__Secure-1PSID": "psid", "__Secure-1PSIDTS": "psidts"}),
            "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts",
        )
        self.assertIsNot(gpt_strategy, grok_strategy)
        self.assertIsNot(grok_strategy, gemini_strategy)

    def test_registry_provider_definitions_expose_capabilities_and_models(self) -> None:
        definitions = provider_registry.provider_definitions()

        self.assertEqual(set(definitions), {GPT_PROVIDER, GROK_PROVIDER, GEMINI_PROVIDER})
        self.assertIn("image", definitions[GPT_PROVIDER].capabilities)
        self.assertIn("image_edit", definitions[GROK_PROVIDER].capabilities)
        self.assertEqual(definitions[GEMINI_PROVIDER].capabilities, frozenset({"chat"}))
        self.assertTrue(any(spec.provider == GPT_PROVIDER for spec in definitions[GPT_PROVIDER].model_specs))
        self.assertTrue(any(spec.provider == GROK_PROVIDER for spec in definitions[GROK_PROVIDER].model_specs))
        self.assertTrue(any(spec.provider == GEMINI_PROVIDER for spec in definitions[GEMINI_PROVIDER].model_specs))

    def test_gemini_account_deletes_by_sanitized_account_id_identifier(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {
                "__Secure-1PSID": "psid-a",
                "__Secure-1PSIDTS": "psidts-a",
                "provider": "gemini",
                "account_id": "gemini-account-a",
            },
            {
                "__Secure-1PSID": "psid-b",
                "__Secure-1PSIDTS": "psidts-b",
                "provider": "gemini",
                "account_id": "gemini-account-b",
            },
        ])

        result = service.delete_accounts([], provider=GEMINI_PROVIDER, identifiers=[{"account_id": "gemini-account-a"}])

        self.assertEqual(result["removed"], 1)
        remaining = service.list_accounts(provider=GEMINI_PROVIDER)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["account_id"], "gemini-account-b")
        self.assertEqual(remaining[0]["access_token"], "__Secure-1PSID=psid-b; __Secure-1PSIDTS=psidts-b")

    def test_gemini_account_deletes_by_sanitized_row_id_identifier_without_account_id(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {
                "__Secure-1PSID": "psid-row",
                "__Secure-1PSIDTS": "psidts-row",
                "provider": "gemini",
            },
        ])
        [account] = service.list_accounts(provider=GEMINI_PROVIDER)
        row_id = provider_registry.account_strategy(GEMINI_PROVIDER).sanitize_account(account)["row_id"]

        result = service.delete_accounts([], provider=GEMINI_PROVIDER, identifiers=[{"row_id": row_id}])

        self.assertEqual(result["removed"], 1)
        self.assertEqual(service.list_accounts(provider=GEMINI_PROVIDER), [])

    def test_grok_account_deletes_by_sanitized_row_id_identifier(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {"access_token": "sso=grok-row-token", "provider": "grok"},
        ])
        [account] = service.list_accounts(provider=GROK_PROVIDER)
        row_id = provider_registry.account_strategy(GROK_PROVIDER).sanitize_account(account)["row_id"]

        result = service.delete_accounts([], provider=GROK_PROVIDER, identifiers=[{"row_id": row_id}])

        self.assertEqual(result["removed"], 1)
        self.assertEqual(service.list_accounts(provider=GROK_PROVIDER), [])

    def test_gemini_account_identifier_delete_is_provider_scoped(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([
            {
                "__Secure-1PSID": "psid-a",
                "__Secure-1PSIDTS": "psidts-a",
                "provider": "gemini",
                "account_id": "shared-account-id",
            },
            {
                "access_token": "gpt-token",
                "provider": "gpt",
                "account_id": "shared-account-id",
            },
        ])

        result = service.delete_accounts([], provider=GEMINI_PROVIDER, identifiers=[{"account_id": "shared-account-id"}])

        self.assertEqual(result["removed"], 1)
        self.assertEqual(service.list_tokens(provider=GPT_PROVIDER), ["gpt-token"])
        self.assertEqual(service.list_accounts(provider=GEMINI_PROVIDER), [])

    def test_gpt_account_delete_by_token_still_ignores_identifier_only_payload(self) -> None:
        service = AccountService(MemoryStorage())
        service.add_account_items([{"access_token": "gpt-token", "provider": "gpt", "account_id": "gpt-account"}])

        result = service.delete_accounts([], provider=GPT_PROVIDER, identifiers=[{"account_id": "gpt-account"}])

        self.assertEqual(result["removed"], 0)
        self.assertEqual(service.list_tokens(provider=GPT_PROVIDER), ["gpt-token"])

    def test_account_service_uses_registry_account_strategy(self) -> None:
        service = AccountService(MemoryStorage())
        strategy = Mock(wraps=provider_registry.account_strategy(GPT_PROVIDER))

        with patch.object(account_service_module, "account_strategy", return_value=strategy) as patched:
            result = service.add_account_items([{"accessToken": "gpt-token", "provider": "gpt"}])

        self.assertEqual(result["added"], 1)
        patched.assert_called()
        self.assertEqual(strategy.normalize_access_token.call_count, 2)



if __name__ == "__main__":
    unittest.main()
