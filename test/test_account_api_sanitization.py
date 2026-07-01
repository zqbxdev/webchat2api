from __future__ import annotations

import sys
import unittest
from typing import Any, cast
from unittest import mock

from test.optional_stubs import StubHTTPException, install_curl_cffi_stub, install_fastapi_stubs, install_pil_stub, install_pybase64_stub, install_pydantic_stub, install_starlette_stub, install_tiktoken_stub

install_curl_cffi_stub()
install_fastapi_stubs()
install_pil_stub()
install_pybase64_stub()
install_pydantic_stub()
install_starlette_stub()
install_tiktoken_stub()

FastAPI = cast(Any, getattr(sys.modules["fastapi"], "FastAPI"))
TestClient = cast(Any, getattr(sys.modules["fastapi.testclient"], "TestClient"))

import api.accounts as accounts_module
from services.account_service import AccountService
from services.models import GROK_PROVIDER
from services.storage.base import StorageBackend


GEMINI_TOKEN = "__Secure-1PSID=psid; __Secure-1PSIDTS=psidts"
AUTH_HEADERS = {"Authorization": "Bearer webchat2api"}
SECRET_KEYS = ("access_token", "cookies", "__Secure-1PSID", "__Secure-1PSIDTS")
GEMINI_SECRET_FRAGMENTS = (GEMINI_TOKEN, "__Secure-1PSID=psid", "__Secure-1PSIDTS=psidts")
GPT_TOKEN = "gpt-token-for-admin-operations"
GROK_TOKEN = "grok-token-for-admin-operations"


class FakeAccountService:
    def __init__(self) -> None:
        self.account = {
            "access_token": GEMINI_TOKEN,
            "provider": "gemini",
            "status": "正常",
            "type": "free",
            "account_id": "gemini-account-1",
            "row_id": "gemini-row-1",
            "cookies": {"__Secure-1PSID": "psid", "__Secure-1PSIDTS": "psidts"},
            "__Secure-1PSID": "psid",
            "__Secure-1PSIDTS": "psidts",
        }
        self.deleted: tuple[list[str], str | None, list[dict[str, str]] | None] | None = None

    def list_accounts(self, provider: str | None = None) -> list[dict[str, Any]]:
        if provider and provider != self.account.get("provider"):
            return []
        return [dict(self.account)]

    def list_tokens(self, provider: str | None = None) -> list[str]:
        if provider and provider != self.account.get("provider"):
            return []
        return [GEMINI_TOKEN]

    def add_account_items(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        return {"added": 1, "skipped": 0, "items": self.list_accounts()}

    def add_accounts(self, tokens: list[str]) -> dict[str, Any]:
        return {"added": 1, "skipped": 0, "items": self.list_accounts()}

    def refresh_accounts(
        self,
        access_tokens: list[str],
        provider: str | None = None,
        identifiers: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        return {"refreshed": 0, "errors": [], "items": self.list_accounts(provider=provider)}

    def validate_accounts(
        self,
        access_tokens: list[str],
        provider: str | None = None,
        identifiers: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        return {"checked": 0, "valid": 0, "invalid": 0, "limited": 0, "unverified": 0, "errors": [], "results": [], "items": self.list_accounts(provider=provider)}

    def update_account(self, access_token: str, updates: dict[str, Any], provider: str | None = None) -> dict[str, Any] | None:
        if provider and provider != self.account.get("provider"):
            return None
        self.account.update(updates)
        return dict(self.account)

    def delete_accounts(
        self,
        tokens: list[str],
        provider: str | None = None,
        identifiers: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        self.deleted = (tokens, provider, identifiers)
        return {"removed": 0, "items": self.list_accounts(provider=provider)}

    def delete_limited_accounts(self, provider: str | None = None) -> dict[str, Any]:
        return {"removed": 0, "items": self.list_accounts(provider=provider)}

    def build_export_items(
        self,
        access_tokens: list[str] | None = None,
        provider: str | None = None,
        identifiers: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        return [{"access_token": GEMINI_TOKEN, "sso": ""}]

    @staticmethod
    def build_export_text(items: list[dict[str, str]]) -> str:
        return "\n".join(item["access_token"] for item in items) + "\n"


class GeminiAccountApiSanitizationTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.modules.pop("services.providers.grok", None)
        self.account_service = FakeAccountService()
        self.patchers = [
            mock.patch.object(accounts_module, "account_service", self.account_service),
            mock.patch.object(accounts_module, "require_admin", lambda authorization: {"role": "admin"}),
            mock.patch("services.log_service.log_service.add"),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        app = FastAPI()
        app.include_router(accounts_module.create_router())
        self.client = TestClient(app)

    def assert_no_gemini_secrets(self, payload: object) -> None:
        text = str(payload)
        for secret in GEMINI_SECRET_FRAGMENTS:
            self.assertNotIn(secret, text)
        if isinstance(payload, dict):
            for key in SECRET_KEYS:
                self.assertNotIn(key, payload)
            self.assertTrue(payload.get("has_gemini_session"))

    def test_get_accounts_sanitizes_gemini_credentials(self) -> None:
        response = self.client.get("/api/accounts", headers=AUTH_HEADERS)

        self.assertEqual(response.status_code, 200)
        body = cast(dict[str, Any], response.json())
        self.assert_no_gemini_secrets(body["items"][0])

    def test_gpt_keeps_access_token_for_admin_identifier_and_grok_redacts_it(self) -> None:
        gpt_sanitized = accounts_module.sanitize_account({
            "provider": "gpt",
            "access_token": GPT_TOKEN,
            "status": "正常",
            "type": "free",
        })
        grok_sanitized = accounts_module.sanitize_account({
            "provider": "grok",
            "access_token": GROK_TOKEN,
            "status": "正常",
            "type": "free",
        })

        self.assertEqual(gpt_sanitized["access_token"], GPT_TOKEN)
        self.assertNotIn("access_token", grok_sanitized)
        self.assertTrue(grok_sanitized["has_access_token"])
        self.assertNotIn(GROK_TOKEN, str(grok_sanitized))

    def test_create_accounts_sanitizes_gemini_credentials(self) -> None:
        response = self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "gemini", "accounts": [{"__Secure-1PSID": "psid", "__Secure-1PSIDTS": "psidts"}]},
        )

        self.assertEqual(response.status_code, 200)
        body = cast(dict[str, Any], response.json())
        self.assert_no_gemini_secrets(body["items"][0])

    def test_create_accounts_rejects_gemini_token_import(self) -> None:
        response = self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "gemini", "tokens": [GEMINI_TOKEN]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("双字段", str(response.json()))

    def test_create_accounts_rejects_gemini_access_token_payload(self) -> None:
        response = self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"accounts": [{"provider": "gemini", "access_token": GEMINI_TOKEN}]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("不支持完整 Cookie", str(response.json()))

    def test_create_accounts_rejects_gemini_named_cookie_values(self) -> None:
        response = self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "gemini", "accounts": [{"__Secure-1PSID": "__Secure-1PSID=psid", "__Secure-1PSIDTS": "psidts"}]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("等号右侧", str(response.json()))

    def test_create_accounts_rejects_mixed_provider_gemini_payload(self) -> None:
        response = self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "gemini", "accounts": [{"provider": "gpt", "__Secure-1PSID": "psid", "__Secure-1PSIDTS": "psidts"}]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("不能混用", str(response.json()))

    def test_refresh_accounts_sanitizes_gemini_credentials(self) -> None:
        response = self.client.post("/api/accounts/refresh", headers=AUTH_HEADERS, json={"access_tokens": [GEMINI_TOKEN]})

        self.assertEqual(response.status_code, 200)
        body = cast(dict[str, Any], response.json())
        self.assert_no_gemini_secrets(body["items"][0])

    def test_update_account_sanitizes_gemini_credentials(self) -> None:
        response = self.client.post(
            "/api/accounts/update",
            headers=AUTH_HEADERS,
            json={"access_token": GEMINI_TOKEN, "status": "正常"},
        )

        self.assertEqual(response.status_code, 200)
        body = cast(dict[str, Any], response.json())
        self.assert_no_gemini_secrets(body["item"])
        self.assert_no_gemini_secrets(body["items"][0])

    def test_provider_filter_is_forwarded_to_get_accounts(self) -> None:
        response = self.client.get("/api/accounts?provider=gpt", headers=AUTH_HEADERS)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["items"], [])

    def test_update_account_target_provider_does_not_match_other_provider(self) -> None:
        response = self.client.post(
            "/api/accounts/update",
            headers=AUTH_HEADERS,
            json={"access_token": GEMINI_TOKEN, "target_provider": "gpt", "status": "禁用"},
        )

        self.assertEqual(response.status_code, 404)

    def test_delete_accounts_forwards_provider_and_sanitizes_gemini_credentials(self) -> None:
        response = self.client._request(
            "DELETE",
            "/api/accounts",
            headers=AUTH_HEADERS,
            json_data={"provider": "gemini", "tokens": [GEMINI_TOKEN]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.account_service.deleted, ([GEMINI_TOKEN], "gemini", []))
        body = cast(dict[str, Any], response.json())
        self.assert_no_gemini_secrets(body["items"][0])

    def test_delete_accounts_accepts_sanitized_gemini_account_identifier(self) -> None:
        response = self.client._request(
            "DELETE",
            "/api/accounts",
            headers=AUTH_HEADERS,
            json_data={"provider": "gemini", "identifiers": [{"account_id": "gemini-account-1"}]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.account_service.deleted, ([], "gemini", [{"account_id": "gemini-account-1"}]))
        body = cast(dict[str, Any], response.json())
        self.assert_no_gemini_secrets(body["items"][0])
        self.assertEqual(body["items"][0]["account_id"], "gemini-account-1")

    def test_delete_accounts_accepts_sanitized_gemini_row_identifier(self) -> None:
        response = self.client._request(
            "DELETE",
            "/api/accounts",
            headers=AUTH_HEADERS,
            json_data={"provider": "gemini", "identifiers": [{"row_id": "gemini-row-1"}]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.account_service.deleted, ([], "gemini", [{"row_id": "gemini-row-1"}]))
        body = cast(dict[str, Any], response.json())
        self.assert_no_gemini_secrets(body["items"][0])

    def test_export_accounts_keeps_gemini_cookie_header(self) -> None:
        items = self.account_service.build_export_items(provider="gemini")
        text = self.account_service.build_export_text(items)

        self.assertIn(GEMINI_TOKEN, text)


class MemoryStorage(StorageBackend):
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

    def load_settings(self) -> dict[str, Any]:
        return {}

    def save_settings(self, settings: dict[str, Any]) -> None:
        pass

    def load_proxy_pool(self) -> list[dict[str, Any]]:
        return []

    def save_proxy_pool(self, items: list[dict[str, Any]]) -> None:
        pass

    def health_check(self) -> dict[str, Any]:
        return {"ok": True}

    def get_backend_info(self) -> dict[str, Any]:
        return {"type": "memory"}


class GrokAccountApiImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.account_service = AccountService(MemoryStorage())
        self.refreshed: list[tuple[list[str], str | None]] = []
        original_refresh = self.account_service.refresh_accounts

        def refresh_accounts(
            access_tokens: list[str],
            provider: str | None = None,
            identifiers: list[dict[str, str]] | None = None,
        ) -> dict[str, Any]:
            self.refreshed.append((list(access_tokens), provider))
            return original_refresh(access_tokens, provider=provider, identifiers=identifiers)

        self.account_service.refresh_accounts = refresh_accounts  # type: ignore[method-assign]
        self.patchers = [
            mock.patch.object(accounts_module, "account_service", self.account_service),
            mock.patch.object(accounts_module, "require_admin", lambda authorization: {"role": "admin"}),
            mock.patch("services.log_service.log_service.add"),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        app = FastAPI()
        app.include_router(accounts_module.create_router())
        self.client = TestClient(app)


    def assert_grok_items_redacted(self, items: list[dict[str, Any]]) -> None:
        self.assertTrue(items)
        self.assertTrue(all(item["has_access_token"] for item in items))
        self.assertTrue(all(item["has_sso"] for item in items))
        text = str(items)
        for secret in (
            "eyJhbGciOiJub25lIn0.eyJzdWIiOiJncm9rIn0",
            "eyJhbGciOiJub25lIn0.eyJzdWIiOiJjb29raWUifQ",
            "eyJhbGciOiJub25lIn0.eyJzdWIiOiJjb29raWUtZGljdCJ9",
            "eyJhbGciOiJub25lIn0.eyJzdWIiOiJpbnZhbGlkIn0",
        ):
            self.assertNotIn(secret, text)
        for item in items:
            self.assertNotIn("access_token", item)
            self.assertNotIn("sso", item)
            self.assertNotIn("raw_sso", item)
            self.assertNotIn("sso_token", item)
            self.assertTrue(item.get("row_id"))

    def test_create_accounts_rejects_grok_account_payloads(self) -> None:
        response = self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={
                "provider": "grok",
                "accounts": [
                    {"raw_sso": "dev-raw-sso-token", "tier": "premium", "capabilities": "chat,heavy"},
                    {"sso_token": "sso=dev-cookie-sso-token; other=value"},
                    {"cookies": {"sso": "dev-cookie-dict-token", "sso-rw": "ignored"}},
                    {"access_token": "unsafe-bare-token"},
                ],
            },
        )

        self.assertEqual(response.status_code, 400)
        message = str(response.json())
        self.assertIn("裸 SSO 值", message)
        self.assertIn("sso=<值>", message)
        self.assertIn("accounts", message)
        self.assertIn("JSON", message)
        self.assertEqual(self.account_service.list_accounts(provider=GROK_PROVIDER), [])

    def test_grok_sso_import_list_select_export_delete_uses_consistent_token_shape(self) -> None:
        sso_token = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJncm9rIn0"
        response = self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "grok", "tokens": [sso_token, f"sso={sso_token}"]},
        )

        self.assertEqual(response.status_code, 200)
        body = cast(dict[str, Any], response.json())
        self.assertEqual(body["added"], 1)
        self.assertEqual(body["skipped"], 0)
        self.assert_grok_items_redacted(body["items"])
        self.assertEqual(self.account_service.list_tokens(provider=GROK_PROVIDER), [sso_token])
        storage = cast(MemoryStorage, self.account_service.storage)
        [stored] = storage.accounts
        self.assertEqual(stored["access_token"], sso_token)
        for alias_key in ("sso", "raw_sso", "sso_token", "cookie", "token", "cookies"):
            self.assertNotIn(alias_key, stored)
        self.assertEqual(self.account_service.get_text_access_token(provider=GROK_PROVIDER), sso_token)

        list_response = self.client.get("/api/accounts?provider=grok", headers=AUTH_HEADERS)
        self.assertEqual(list_response.status_code, 200)
        list_body = cast(dict[str, Any], list_response.json())
        self.assert_grok_items_redacted(list_body["items"])
        row_id = str(list_body["items"][0]["row_id"])

        row_export_response = self.client.post(
            "/api/accounts/export",
            headers=AUTH_HEADERS,
            json={"provider": "grok", "identifiers": [{"row_id": row_id}]},
        )
        self.assertEqual(row_export_response.status_code, 200)
        self.assertEqual(row_export_response.text, f"{sso_token}\n")

        for selector in (sso_token, f"sso={sso_token}"):
            export_response = self.client.post(
                "/api/accounts/export",
                headers=AUTH_HEADERS,
                json={"provider": "grok", "access_tokens": [selector]},
            )
            self.assertEqual(export_response.status_code, 200)
            self.assertEqual(export_response.text, f"{sso_token}\n")

        delete_response = self.client.delete(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "grok", "tokens": [f"sso={sso_token}"]},
        )
        self.assertEqual(delete_response.status_code, 200)
        delete_body = cast(dict[str, Any], delete_response.json())
        self.assertEqual(delete_body["removed"], 1)
        self.assertEqual(delete_body["items"], [])
        self.assertEqual(self.account_service.list_tokens(provider=GROK_PROVIDER), [])

        reimport_response = self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "grok", "tokens": [f"sso={sso_token}"]},
        )
        self.assertEqual(reimport_response.status_code, 200)
        self.assertEqual(self.account_service.list_tokens(provider=GROK_PROVIDER), [sso_token])

        bare_delete_response = self.client.delete(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "grok", "tokens": [sso_token]},
        )
        self.assertEqual(bare_delete_response.status_code, 200)
        bare_delete_body = cast(dict[str, Any], bare_delete_response.json())
        self.assertEqual(bare_delete_body["removed"], 1)
        self.assertEqual(self.account_service.list_tokens(provider=GROK_PROVIDER), [])

        self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "grok", "tokens": [f"sso={sso_token}"]},
        )
        row_delete_response = self.client.delete(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "grok", "identifiers": [{"row_id": row_id}]},
        )
        self.assertEqual(row_delete_response.status_code, 200)
        row_delete_body = cast(dict[str, Any], row_delete_response.json())
        self.assertEqual(row_delete_body["removed"], 1)
        self.assertEqual(self.account_service.list_tokens(provider=GROK_PROVIDER), [])

    def test_create_accounts_imports_grok_sso_token_list_payload(self) -> None:
        response = self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "grok", "tokens": ["sso=eyJhbGciOiJub25lIn0.eyJzdWIiOiJsaXN0In0", "eyJhbGciOiJub25lIn0.eyJzdWIiOiJwdXJlIn0"]},
        )

        self.assertEqual(response.status_code, 200)
        body = cast(dict[str, Any], response.json())
        self.assertEqual(body["added"], 2)
        self.assert_grok_items_redacted(body["items"])
        self.assertEqual(self.account_service.list_tokens(provider=GROK_PROVIDER), ["eyJhbGciOiJub25lIn0.eyJzdWIiOiJsaXN0In0", "eyJhbGciOiJub25lIn0.eyJzdWIiOiJwdXJlIn0"])
        accounts = self.account_service.list_accounts(provider=GROK_PROVIDER)
        self.assertTrue(all(account["access_token"] for account in accounts))

    def test_create_accounts_rejects_grok_non_sso_cookie_list_payload(self) -> None:
        for token in ("sso-rw=unsafe-rw-token", "other=unsafe-token", "sso=unsafe-token; sso-rw=unsafe-rw-token"):
            with self.subTest(token=token):
                response = self.client.post(
                    "/api/accounts",
                    headers=AUTH_HEADERS,
                    json={"provider": "grok", "tokens": [token]},
                )

                self.assertEqual(response.status_code, 400)
                message = str(response.json())
                self.assertIn("不支持 sso-rw", message)
                self.assertIn("完整 Cookie header", message)
                self.assertEqual(self.account_service.list_accounts(provider=GROK_PROVIDER), [])
                self.assertEqual(self.refreshed, [])

    def test_create_accounts_keeps_grok_imports_when_validation_would_remove_them(self) -> None:
        def destructive_refresh(
            access_tokens: list[str],
            provider: str | None = None,
            identifiers: list[dict[str, str]] | None = None,
        ) -> dict[str, Any]:
            self.refreshed.append((list(access_tokens), provider))
            for access_token in access_tokens:
                self.account_service.remove_invalid_token(access_token, "refresh_accounts")
            return {
                "refreshed": 0,
                "errors": [{"token": token, "error": "Grok app-chat authentication failed (HTTP 401)"} for token in access_tokens],
                "items": self.account_service.list_accounts(provider=provider),
            }

        self.account_service.refresh_accounts = destructive_refresh  # type: ignore[method-assign]

        response = self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "grok", "tokens": ["sso=eyJhbGciOiJub25lIn0.eyJzdWIiOiJyYXcxIn0", "sso=eyJhbGciOiJub25lIn0.eyJzdWIiOiJyYXcyIn0", "sso=eyJhbGciOiJub25lIn0.eyJzdWIiOiJyYXczIn0"]},
        )

        self.assertEqual(response.status_code, 200)
        body = cast(dict[str, Any], response.json())
        self.assertEqual(body["added"], 3)
        self.assertEqual(body["refreshed"], 0)
        self.assertEqual(body["errors"], [])
        self.assertEqual(self.refreshed, [])
        self.assert_grok_items_redacted(body["items"])
        self.assertEqual(self.account_service.list_tokens(provider="grok"), [
            "eyJhbGciOiJub25lIn0.eyJzdWIiOiJyYXcxIn0",
            "eyJhbGciOiJub25lIn0.eyJzdWIiOiJyYXcyIn0",
            "eyJhbGciOiJub25lIn0.eyJzdWIiOiJyYXczIn0",
        ])

        list_response = self.client.get("/api/accounts?provider=grok", headers=AUTH_HEADERS)
        self.assertEqual(list_response.status_code, 200)
        list_body = cast(dict[str, Any], list_response.json())
        self.assert_grok_items_redacted(list_body["items"])
        self.assertEqual(self.account_service.list_tokens(provider="grok"), [
            "eyJhbGciOiJub25lIn0.eyJzdWIiOiJyYXcxIn0",
            "eyJhbGciOiJub25lIn0.eyJzdWIiOiJyYXcyIn0",
            "eyJhbGciOiJub25lIn0.eyJzdWIiOiJyYXczIn0",
        ])

    def test_manual_refresh_can_still_apply_grok_invalid_token_policy(self) -> None:
        response = self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "grok", "tokens": ["sso=eyJhbGciOiJub25lIn0.eyJzdWIiOiJpbnZhbGlkIn0"]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.account_service.list_accounts(provider=GROK_PROVIDER)), 1)

        def destructive_refresh(
            access_tokens: list[str],
            provider: str | None = None,
            identifiers: list[dict[str, str]] | None = None,
        ) -> dict[str, Any]:
            self.refreshed.append((list(access_tokens), provider))
            for access_token in access_tokens:
                self.account_service.remove_invalid_token(access_token, "refresh_accounts")
            return {
                "refreshed": 0,
                "errors": [{"token": token, "error": "Grok app-chat authentication failed (HTTP 401)"} for token in access_tokens],
                "items": self.account_service.list_accounts(provider=provider),
            }

        self.account_service.refresh_accounts = destructive_refresh  # type: ignore[method-assign]
        refresh_response = self.client.post(
            "/api/accounts/refresh",
            headers=AUTH_HEADERS,
            json={"provider": "grok", "access_tokens": ["eyJhbGciOiJub25lIn0.eyJzdWIiOiJpbnZhbGlkIn0"]},
        )

        self.assertEqual(refresh_response.status_code, 200)
        self.assertEqual(self.refreshed, [(["eyJhbGciOiJub25lIn0.eyJzdWIiOiJpbnZhbGlkIn0"], "grok")])
        self.assertEqual(self.account_service.list_accounts(provider=GROK_PROVIDER), [])
        refresh_body = cast(dict[str, Any], refresh_response.json())
        self.assertEqual(refresh_body["items"], [])

    def test_create_accounts_rejects_grok_payloads_without_safe_sso_evidence(self) -> None:
        response = self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={
                "provider": "grok",
                "accounts": [
                    {"access_token": "unsafe-bare-token"},
                    {"token": "unsafe-token-field"},
                    {"cookies": {"sso-rw": "unsafe-rw-token"}},
                ],
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("JSON", str(response.json()))
        self.assertEqual(self.account_service.list_accounts(provider=GROK_PROVIDER), [])
        self.assertEqual(self.refreshed, [])

    def test_create_accounts_does_not_treat_gpt_sso_alias_as_token(self) -> None:
        response = self.client.post(
            "/api/accounts",
            headers=AUTH_HEADERS,
            json={"provider": "gpt", "accounts": [{"raw_sso": "not-a-gpt-token"}]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": "tokens is required"})
        self.assertEqual(self.account_service.list_accounts(provider="gpt"), [])
        self.assertEqual(self.refreshed, [])


class AccountValidateApiSanitizationTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.modules.pop("services.providers.grok", None)
        self.account_service = FakeAccountService()
        self.patchers = [
            mock.patch.object(accounts_module, "account_service", self.account_service),
            mock.patch.object(accounts_module, "require_admin", lambda authorization: {"role": "admin"}),
            mock.patch("services.log_service.log_service.add"),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        app = FastAPI()
        app.include_router(accounts_module.create_router())
        self.client = TestClient(app)

    def test_validate_accounts_rejects_unsupported_provider(self) -> None:
        response = self.client.post(
            "/api/accounts/validate",
            headers=AUTH_HEADERS,
            json={"provider": "gemini", "access_tokens": [GEMINI_TOKEN]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported", str(response.json()).lower())
        self.assertNotIn(GEMINI_TOKEN, str(response.json()))
        self.assertIs(accounts_module.HTTPException, StubHTTPException)

    def test_validate_accounts_sanitizes_grok_result(self) -> None:
        self.account_service.account = {
            "access_token": GROK_TOKEN,
            "provider": "grok",
            "status": "正常",
            "type": "free",
            "account_id": "grok-account-1",
            "raw_sso": "raw-sso-token",
        }

        def validate_accounts(
            access_tokens: list[str],
            provider: str | None = None,
            identifiers: list[dict[str, str]] | None = None,
        ) -> dict[str, Any]:
            return {
                "checked": 1,
                "valid": 0,
                "invalid": 1,
                "limited": 0,
                "unverified": 0,
                "errors": [{"token": "token:redacted", "error": "Grok app-chat credential invalid"}],
                "results": [{"token": "token:redacted", "status": "invalid", "message": "Grok app-chat credential invalid"}],
                "items": [{
                    "access_token": GROK_TOKEN,
                    "provider": "grok",
                    "status": "异常",
                    "raw_sso": "raw-sso-token",
                    "account_id": "grok-account-1",
                }],
            }

        self.account_service.validate_accounts = validate_accounts  # type: ignore[method-assign]
        response = self.client.post(
            "/api/accounts/validate",
            headers=AUTH_HEADERS,
            json={"provider": "grok", "access_tokens": [GROK_TOKEN]},
        )

        self.assertEqual(response.status_code, 200)
        body = cast(dict[str, Any], response.json())
        self.assertEqual(body["checked"], 1)
        self.assertEqual(body["invalid"], 1)
        self.assertNotIn(GROK_TOKEN, str(body))
        self.assertNotIn("raw-sso-token", str(body))
        self.assertNotIn("access_token", body["items"][0])
        self.assertTrue(body["items"][0]["has_access_token"])


if __name__ == "__main__":
    unittest.main()
