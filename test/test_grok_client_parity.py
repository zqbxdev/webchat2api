from __future__ import annotations

from typing import Any
from types import ModuleType
import sys
import types
import unittest
from unittest import mock

from test.optional_stubs import install_curl_cffi_stub, install_fastapi_stubs, install_pil_stub, install_pybase64_stub, install_pydantic_stub, install_starlette_stub, install_tiktoken_stub

install_curl_cffi_stub()
install_fastapi_stubs()
install_pil_stub()
install_pybase64_stub()
install_pydantic_stub()
install_starlette_stub()
install_tiktoken_stub()

from services.models import resolve_model
from services.providers.grok import client as grok


def _account_service_module(account_service: object) -> ModuleType:
    module = ModuleType("services.account_service")
    setattr(module, "account_service", account_service)
    return module


class GrokClientParityTests(unittest.TestCase):
    def test_rate_limit_payload_includes_upstream_model_only(self) -> None:
        self.assertEqual(
            grok.build_grok_rate_limits_payload({"modelName": "grok-custom"}),
            {"modelName": "grok-custom"},
        )
        self.assertEqual(grok.build_grok_rate_limits_payload()["modelName"], "auto")

    def test_rate_limit_helpers_extract_nested_remaining_window_and_tier_hints(self) -> None:
        payload = {
            "rateLimits": {
                "text": {"remainingQueries": "17"},
                "limits": {"window_size_seconds": "7200"},
            },
            "account": {"subscriptionTier": "premium"},
        }

        self.assertEqual(grok.extract_grok_rate_limit_remaining(payload), 17)
        self.assertEqual(grok.extract_grok_rate_limit_window_seconds(payload), 7200)
        self.assertEqual(grok.grok_rate_limit_account_hints(payload), {
            "app_chat": True,
            "quota": 17,
            "status": "正常",
            "rate_limit_window_seconds": 7200,
            "tier": "super",
        })

    def test_rate_limit_helpers_use_window_for_safe_basic_tier_hint(self) -> None:
        payload = {"limits": {"remainingTokens": 0, "windowSizeSeconds": 28800}}

        self.assertEqual(grok.grok_rate_limit_account_hints(payload), {
            "app_chat": True,
            "quota": 0,
            "status": "限流",
            "rate_limit_window_seconds": 28800,
            "tier": "basic",
        })

    def test_validate_rate_limits_posts_upstream_payload_and_updates_hints(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())
        posted: dict[str, Any] = {}

        class FakeResponse:
            status_code = 200

            def json(self) -> object:
                return {"limits": {"remainingQueries": 3, "windowSizeSeconds": 7200}}

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                posted.update(kwargs)
                return FakeResponse()

            def close(self) -> None:
                pass

        with (
            mock.patch.dict(sys.modules, {"services.account_service": _account_service_module(account_service)}),
            mock.patch.object(grok.config, "data", {}),
            mock.patch("curl_cffi.requests.Session", FakeSession),
        ):
            client = grok.GrokAppChatClient("secret-token", {"modelName": "grok-test"})
            self.assertEqual(client.validate_rate_limits(), {"limits": {"remainingQueries": 3, "windowSizeSeconds": 7200}})

        self.assertEqual(posted["json"], {"modelName": "grok-test"})
        account_service.update_account.assert_called_once_with(
            "secret-token",
            {
                "app_chat": True,
                "quota": 3,
                "status": "正常",
                "rate_limit_window_seconds": 7200,
                "tier": "super",
                "quota_console": {"remaining": 3, "window_seconds": 7200, "tier": "super"},
            },
        )

    def test_validate_rate_limits_cloudflare_html_is_check_unavailable(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())

        class FakeResponse:
            status_code = 403
            text = "<html><title>Just a moment...</title><script src='/cdn-cgi/challenge-platform/h/b'></script></html>"

            def json(self) -> object:
                raise ValueError("not json")

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with (
            mock.patch.dict(sys.modules, {"services.account_service": _account_service_module(account_service)}),
            mock.patch.object(grok.config, "data", {}),
            mock.patch("curl_cffi.requests.Session", FakeSession),
        ):
            client = grok.GrokAppChatClient("secret-token")
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                client.validate_rate_limits()

        self.assertEqual(ctx.exception.code, "cloudflare_challenge")
        self.assertTrue(ctx.exception.is_check_unavailable)
        account_service.update_account.assert_not_called()

    def test_validate_rate_limits_unknown_403_is_check_unavailable(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())

        class FakeResponse:
            status_code = 403
            text = '{"message":"request blocked"}'

            def json(self) -> object:
                return {"message": "request blocked"}

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with (
            mock.patch.dict(sys.modules, {"services.account_service": _account_service_module(account_service)}),
            mock.patch.object(grok.config, "data", {}),
            mock.patch("curl_cffi.requests.Session", FakeSession),
        ):
            client = grok.GrokAppChatClient("secret-token")
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                client.validate_rate_limits()

        self.assertEqual(ctx.exception.code, "rate_limit_check_unavailable")
        self.assertTrue(ctx.exception.is_check_unavailable)
        account_service.update_account.assert_not_called()

    def test_validate_rate_limits_bare_401_is_check_unavailable(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())

        class FakeResponse:
            status_code = 401
            text = ""

            def json(self) -> object:
                raise ValueError("not json")

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with (
            mock.patch.dict(sys.modules, {"services.account_service": _account_service_module(account_service)}),
            mock.patch.object(grok.config, "data", {}),
            mock.patch("curl_cffi.requests.Session", FakeSession),
        ):
            client = grok.GrokAppChatClient("secret-token")
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                client.validate_rate_limits()

        self.assertEqual(ctx.exception.code, "rate_limit_check_unavailable")
        self.assertTrue(ctx.exception.is_check_unavailable)
        account_service.update_account.assert_not_called()

    def test_validate_rate_limits_confirmed_invalid_marker_is_auth_failure(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())

        class FakeResponse:
            status_code = 403
            text = '{"code":"invalid-credentials"}'

            def json(self) -> object:
                return {"code": "invalid-credentials"}

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with (
            mock.patch.dict(sys.modules, {"services.account_service": _account_service_module(account_service)}),
            mock.patch.object(grok.config, "data", {}),
            mock.patch("curl_cffi.requests.Session", FakeSession),
        ):
            client = grok.GrokAppChatClient("secret-token")
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                client.validate_rate_limits()

        self.assertEqual(ctx.exception.code, "authentication_failed")
        account_service.update_account.assert_called_once_with("secret-token", {"status": "异常"})

    def test_validate_rate_limits_generic_auth_marker_is_check_unavailable(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())

        class FakeResponse:
            status_code = 403
            text = '{"code":"authentication_failed","message":"unauthenticated"}'

            def json(self) -> object:
                return {"code": "authentication_failed", "message": "unauthenticated"}

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with (
            mock.patch.dict(sys.modules, {"services.account_service": _account_service_module(account_service)}),
            mock.patch.object(grok.config, "data", {}),
            mock.patch("curl_cffi.requests.Session", FakeSession),
        ):
            client = grok.GrokAppChatClient("secret-token")
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                client.validate_rate_limits()

        self.assertEqual(ctx.exception.code, "rate_limit_check_unavailable")
        self.assertTrue(ctx.exception.is_check_unavailable)
        account_service.update_account.assert_not_called()

    def test_validate_rate_limits_429_is_rate_limited(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())

        class FakeResponse:
            status_code = 429
            text = '{"message":"rate limited"}'

            def json(self) -> object:
                return {"message": "rate limited"}

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with (
            mock.patch.dict(sys.modules, {"services.account_service": _account_service_module(account_service)}),
            mock.patch.object(grok.config, "data", {}),
            mock.patch("curl_cffi.requests.Session", FakeSession),
        ):
            client = grok.GrokAppChatClient("secret-token")
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                client.validate_rate_limits()

        self.assertEqual(ctx.exception.code, "rate_limit_exceeded")
        account_service.update_account.assert_called_once_with("secret-token", {"status": "限流"})

    def test_app_chat_line_events_handles_raw_json_data_prefixes_embedded_blocks_and_done(self) -> None:
        events = list(grok.app_chat_line_events([
            b'{"token":"raw"}',
            b'data: {"token":"first"}data: {"token":"second"}',
            b'data: [DONE]',
            b'data: {"result":',
            b'data: {"response":{"token":"split"}}}',
            b'',
        ]))

        self.assertEqual(events, [
            {"token": "raw"},
            {"token": "first"},
            {"token": "second"},
            {"result": {"response": {"token": "split"}}},
        ])

    def test_app_chat_completion_marks_text_used_only_after_successful_iteration(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(return_value="selected-token"),
            get_account=mock.Mock(return_value={"access_token": "selected-token"}),
            mark_text_used=mock.Mock(),
        )
        spec = resolve_model("grok-4.20-heavy")

        def broken_stream() -> Any:
            yield {"result": {"response": {"token": "partial", "messageTag": "final"}}}
            raise RuntimeError("stream broke")

        with (
            mock.patch.dict(sys.modules, {"services.account_service": _account_service_module(account_service)}),
            mock.patch.object(grok, "GrokAppChatClient") as client_class,
        ):
            client = client_class.return_value.__enter__.return_value
            client.stream_events.return_value = broken_stream()
            events = grok.app_chat_completion_events({}, spec, [{"role": "user", "content": "Hello"}])
            self.assertEqual(next(events), {"result": {"response": {"token": "partial", "messageTag": "final"}}})
            with self.assertRaises(RuntimeError):
                next(events)

        account_service.mark_text_used.assert_not_called()

    def test_app_chat_completion_retries_retryable_errors_with_next_token(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(side_effect=["first-token", "second-token"]),
            get_account=mock.Mock(side_effect=[{"access_token": "first-token"}, {"access_token": "second-token"}]),
            mark_text_used=mock.Mock(),
        )
        spec = resolve_model("grok-4.20-heavy")

        class FailingClient:
            def __init__(self, access_token: str, account: dict[str, Any] | None = None) -> None:
                self.access_token = access_token

            def __enter__(self) -> "FailingClient":
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                return None

            def stream_events(self, payload: dict[str, Any]) -> Any:
                if self.access_token == "first-token":
                    raise grok.GrokConsoleError("limited", 429, 429, "rate_limit_exceeded")
                yield {"result": {"response": {"token": "ok", "isSoftStop": True}}}

        with (
            mock.patch.dict(sys.modules, {"services.account_service": _account_service_module(account_service)}),
            mock.patch.object(grok, "GrokAppChatClient", FailingClient),
        ):
            events = list(grok.app_chat_completion_events({}, spec, [{"role": "user", "content": "Hello"}]))

        self.assertEqual(events, [{"result": {"response": {"token": "ok", "isSoftStop": True}}}])
        self.assertEqual(account_service.get_grok_app_chat_access_token.call_args_list[1].kwargs["excluded_tokens"], {"first-token"})
        account_service.mark_text_used.assert_called_once_with("second-token")

    def test_app_chat_completion_refreshes_once_before_no_account_failure(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(return_value=""),
            refresh_accounts=mock.Mock(),
            mark_text_used=mock.Mock(),
        )
        spec = resolve_model("grok-4.20-heavy")

        with mock.patch.dict(sys.modules, {"services.account_service": _account_service_module(account_service)}):
            with self.assertRaises(Exception) as context:
                list(grok.app_chat_completion_events({}, spec, [{"role": "user", "content": "Hello"}]))

        self.assertEqual(getattr(context.exception, "status_code", None), 503)
        account_service.refresh_accounts.assert_called_once_with([], provider="grok")
        account_service.mark_text_used.assert_not_called()
    def test_console_chat_completion_marks_console_used_on_success(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_console_access_token=mock.Mock(return_value="console-token"),
            get_account=mock.Mock(return_value={}),
            mark_grok_console_used=mock.Mock(),
        )
        spec = resolve_model("grok-4.20-non-reasoning")

        class FakeConsoleClient:
            def __init__(self, access_token: str, account: Any = None) -> None:
                self.access_token = access_token

            def __enter__(self) -> "FakeConsoleClient":
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                return None

            def create_response(self, payload: dict[str, Any]) -> dict[str, Any]:
                return {"output": [{"content": [{"type": "output_text", "text": "ok"}]}]}

        with (
            mock.patch.dict(sys.modules, {"services.account_service": _account_service_module(account_service)}),
            mock.patch.object(grok, "GrokConsoleClient", FakeConsoleClient),
        ):
            completion = grok.console_chat_completion({}, spec, [{"role": "user", "content": "Hello"}])

        self.assertEqual(completion.content, "ok")
        account_service.mark_grok_console_used.assert_called_once_with("console-token", success=True)

    def test_console_chat_completion_events_marks_console_used_on_success(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_console_access_token=mock.Mock(return_value="console-token"),
            get_account=mock.Mock(return_value={}),
            mark_grok_console_used=mock.Mock(),
        )
        spec = resolve_model("grok-4.20-non-reasoning")

        class FakeConsoleClient:
            def __init__(self, access_token: str, account: Any = None) -> None:
                self.access_token = access_token

            def __enter__(self) -> "FakeConsoleClient":
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                return None

            def stream_response(self, payload: dict[str, Any]) -> Any:
                yield {"type": "response.output_text.delta", "delta": "ok"}

        with (
            mock.patch.dict(sys.modules, {"services.account_service": _account_service_module(account_service)}),
            mock.patch.object(grok, "GrokConsoleClient", FakeConsoleClient),
        ):
            events = list(grok.console_chat_completion_events({}, spec, [{"role": "user", "content": "Hello"}]))

        self.assertEqual(events, [{"type": "response.output_text.delta", "delta": "ok"}])
        account_service.mark_grok_console_used.assert_called_once_with("console-token", success=True)


if __name__ == "__main__":
    unittest.main()
