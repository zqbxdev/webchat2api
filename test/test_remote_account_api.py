from __future__ import annotations

from collections.abc import Mapping
import sys
import unittest
from typing import Any, cast
from unittest import mock

from test.optional_stubs import install_curl_cffi_stub, install_fastapi_stubs, install_pil_stub, install_pybase64_stub, install_pydantic_stub, install_starlette_stub, install_tiktoken_stub

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
from services.remote_account_service import RemoteAccountService
from services.storage.base import StorageBackend


AUTH_HEADERS = {"Authorization": "Bearer webchat2api"}


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


class FakeRemoteAccountConfig:
    def __init__(self) -> None:
        self.sources: list[dict[str, Any]] = []
        self.deleted_ids: list[str] = []

    def list_sources(self) -> list[dict[str, Any]]:
        return list(self.sources)

    def add_source(self, **values: Any) -> dict[str, Any]:
        source = {"id": "source-1", "import_job": None, **values}
        self.sources.append(source)
        return source

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        for source in self.sources:
            if source["id"] == source_id:
                return source
        return None

    def update_source(self, source_id: str, updates: Mapping[str, Any]) -> dict[str, Any] | None:
        source = self.get_source(source_id)
        if source is None:
            return None
        source.update(updates)
        return source

    def delete_source(self, source_id: str) -> bool:
        before = len(self.sources)
        self.sources = [source for source in self.sources if source["id"] != source_id]
        return len(self.sources) < before


class FakeRemoteAccountImportService:
    def __init__(self) -> None:
        self.inject_calls: list[tuple[Any, dict[str, Any]]] = []
        self.sync_calls: list[tuple[Mapping[str, Any], FakeRemoteAccountConfig]] = []

    def inject_payload(self, payload: Any, **kwargs: Any) -> dict[str, Any]:
        self.inject_calls.append((payload, kwargs))
        return {
            "strategy": kwargs.get("strategy", "merge"),
            "source_id": kwargs.get("source_id", ""),
            "source_name": kwargs.get("source_name", ""),
            "total": 1,
            "added": 1,
            "skipped": 0,
            "removed": 0,
        }

    def sync_source(self, source: Mapping[str, Any], config: FakeRemoteAccountConfig) -> dict[str, Any]:
        self.sync_calls.append((source, config))
        if source.get("fail"):
            raise RuntimeError("secret-token bearer-secret account-secret-token raw response body")
        return {"status": "success", "total": 1, "added": 1, "skipped": 0, "removed": 0, "failed": 0, "errors": []}


class RemoteAccountApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = FakeRemoteAccountConfig()
        self.import_service = FakeRemoteAccountImportService()
        self.patchers = [
            mock.patch.object(accounts_module, "remote_account_config", self.config),
            mock.patch.object(accounts_module, "remote_account_import_service", self.import_service),
            mock.patch.object(accounts_module, "require_admin", lambda authorization: {"role": "admin"}),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        app = FastAPI()
        app.include_router(accounts_module.create_router())
        self.client = TestClient(app)

    def test_create_source_sanitizes_secrets(self) -> None:
        response = self.client.post(
            "/api/remote-account/sources",
            headers=AUTH_HEADERS,
            json={
                "name": "Remote",
                "url": "https://example.test/accounts",
                "auth_token": "secret-token",
                "bearer_token": "bearer-secret",
                "provider": "grok",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertNotIn("auth_token", body["source"])
        self.assertNotIn("bearer_token", body["source"])
        self.assertTrue(body["source"]["has_auth_token"])
        self.assertTrue(body["source"]["has_bearer_token"])
        self.assertEqual(body["source"]["provider"], "grok")

    def test_create_source_accepts_gemini_provider(self) -> None:
        response = self.client.post(
            "/api/remote-account/sources",
            headers=AUTH_HEADERS,
            json={"name": "Gemini Remote", "url": "https://example.test/gemini", "provider": "gemini"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source"]["provider"], "gemini")

    def test_direct_inject_passes_payload_and_strategy(self) -> None:
        response = self.client.post(
            "/api/remote-account/inject",
            headers=AUTH_HEADERS,
            json={
                "tokens": ["token-1"],
                "strategy": "merge",
                "source_id": "manual",
                "source_name": "Manual",
                "provider": "gpt",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload, kwargs = self.import_service.inject_calls[0]
        self.assertEqual(payload, {"tokens": ["token-1"]})
        body = response.json()
        body_text = response.text
        self.assertNotIn("items", body)
        self.assertNotIn("access_token", body_text)
        self.assertNotIn("token-1", body_text)
        self.assertNotIn('"auth_token"', body_text)
        self.assertNotIn('"bearer_token"', body_text)
        self.assertEqual(kwargs["strategy"], "merge")
        self.assertEqual(kwargs["source_id"], "manual")
        self.assertEqual(kwargs["source_name"], "Manual")
        self.assertEqual(kwargs["provider_default"], "gpt")

    def test_direct_inject_accepts_gemini_provider(self) -> None:
        response = self.client.post(
            "/api/remote-account/inject",
            headers=AUTH_HEADERS,
            json={"tokens": ["gemini-cookie"], "provider": "gemini"},
        )

        self.assertEqual(response.status_code, 200)
        _, kwargs = self.import_service.inject_calls[-1]
        self.assertEqual(kwargs["provider_default"], "gemini")

    def test_direct_inject_grok_bare_sso_uses_real_remote_service(self) -> None:
        storage = MemoryStorage()
        real_service = RemoteAccountService(AccountService(storage))
        with mock.patch.object(accounts_module, "remote_account_import_service", real_service):
            response = self.client.post(
                "/api/remote-account/inject",
                headers=AUTH_HEADERS,
                json={"tokens": ["bare-sso"], "provider": "grok"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 1)
        self.assertEqual(response.json()["added"], 1)
        self.assertEqual(response.json()["skipped"], 0)
        self.assertEqual(storage.accounts[0]["provider"], GROK_PROVIDER)
        self.assertEqual(storage.accounts[0]["access_token"], "bare-sso")

    def test_direct_inject_grok_sso_prefixed_uses_real_remote_service(self) -> None:
        storage = MemoryStorage()
        real_service = RemoteAccountService(AccountService(storage))
        with mock.patch.object(accounts_module, "remote_account_import_service", real_service):
            response = self.client.post(
                "/api/remote-account/inject",
                headers=AUTH_HEADERS,
                json={"tokens": ["sso=bare-sso"], "provider": "grok"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 1)
        self.assertEqual(response.json()["added"], 1)
        self.assertEqual(response.json()["skipped"], 0)
        self.assertEqual(storage.accounts[0]["provider"], GROK_PROVIDER)
        self.assertEqual(storage.accounts[0]["access_token"], "bare-sso")

    def test_direct_inject_grok_flowpilot_account_uses_real_remote_service(self) -> None:
        storage = MemoryStorage()
        real_service = RemoteAccountService(AccountService(storage))
        with mock.patch.object(accounts_module, "remote_account_import_service", real_service):
            response = self.client.post(
                "/api/remote-account/inject",
                headers=AUTH_HEADERS,
                json={
                    "accounts": [{"token": "flowpilot-sso", "provider": "grok", "type": "sso"}],
                    "strategy": "merge",
                    "source_id": "flowpilot-grok-sso",
                    "source_name": "FlowPilot Grok SSO",
                    "provider": "grok",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 1)
        self.assertEqual(response.json()["added"], 1)
        self.assertEqual(response.json()["skipped"], 0)
        self.assertEqual(storage.accounts[0]["provider"], GROK_PROVIDER)
        self.assertEqual(storage.accounts[0]["access_token"], "flowpilot-sso")

    def test_direct_inject_grok_invalid_shape_returns_400(self) -> None:
        real_service = RemoteAccountService(AccountService(MemoryStorage()))
        with mock.patch.object(accounts_module, "remote_account_import_service", real_service):
            response = self.client.post(
                "/api/remote-account/inject",
                headers=AUTH_HEADERS,
                json={"tokens": ["sso-rw=unsafe-rw-token"], "provider": "grok"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("bare SSO", str(response.json()))

    def test_sync_source_generic_exception_response_is_sanitized(self) -> None:
        self.config.sources.append({
            "id": "source-1",
            "name": "Remote",
            "url": "https://example.test/accounts",
            "auth_token": "secret-token",
            "bearer_token": "bearer-secret",
            "fail": True,
        })

        response = self.client.post("/api/remote-account/sources/source-1/sync", headers=AUTH_HEADERS)

        self.assertEqual(response.status_code, 502)
        body_text = response.text
        self.assertIn("remote account sync failed", body_text)
        self.assertNotIn("secret-token", body_text)
        self.assertNotIn("bearer-secret", body_text)
        self.assertNotIn("account-secret-token", body_text)
        self.assertNotIn("raw response body", body_text)

    def test_sync_source_uses_configured_source(self) -> None:
        self.config.sources.append({"id": "source-1", "name": "Remote", "url": "https://example.test/accounts"})

        response = self.client.post("/api/remote-account/sources/source-1/sync", headers=AUTH_HEADERS)

        self.assertEqual(response.status_code, 200)
        body_text = response.text
        self.assertNotIn("access_token", body_text)
        self.assertNotIn("submitted-token", body_text)
        self.assertNotIn('"auth_token"', body_text)
        self.assertNotIn('"bearer_token"', body_text)
        source, config = self.import_service.sync_calls[0]
        self.assertEqual(source["id"], "source-1")
        self.assertIs(config, self.config)


if __name__ == "__main__":
    unittest.main()
