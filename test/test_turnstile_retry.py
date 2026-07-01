from __future__ import annotations

import types
import unittest
from typing import Any
from unittest import mock

from test.optional_stubs import install_curl_cffi_stub, install_fastapi_stubs, install_pil_stub, install_pybase64_stub, install_tiktoken_stub

install_curl_cffi_stub()
install_fastapi_stubs()
install_pil_stub()
install_pybase64_stub()
install_tiktoken_stub()

from services.openai_backend_api import RetryableTurnstileError
from services.protocol import conversation
from services.providers.gpt import runtime as gpt_runtime


class TurnstileRetryTests(unittest.TestCase):
    def test_retryable_turnstile_error_rotates_to_next_account_without_removing_old_token(self) -> None:
        created_tokens: list[str] = []

        class FakeBackend:
            def __init__(self, access_token: str = "", **kwargs) -> None:
                self.access_token = access_token
                created_tokens.append(access_token)

            def close(self) -> None:
                pass

        def fake_conversation_events(backend: FakeBackend, **kwargs: object):
            if backend.access_token == "first-token":
                raise RetryableTurnstileError("turnstile token generation returned None")
            yield {"type": "conversation.delta", "delta": "second account response"}

        requested_attempts: list[set[str]] = []
        account_service = mock.Mock()

        def get_text_access_token(attempted_tokens: set[str]) -> str:
            requested_attempts.append(set(attempted_tokens))
            return "second-token"

        account_service.get_text_access_token.side_effect = get_text_access_token
        request = conversation.ConversationRequest(
            model="auto",
            messages=[{"role": "user", "content": "hello"}],
        )
        initial_backend: Any = types.SimpleNamespace(access_token="first-token")

        with (
            mock.patch.object(gpt_runtime, "OpenAIBackendAPI", FakeBackend),
            mock.patch.object(gpt_runtime, "conversation_events", fake_conversation_events),
            mock.patch.object(gpt_runtime, "account_service", account_service),
            mock.patch.object(gpt_runtime, "_account_proxy", return_value=""),
        ):
            deltas = list(conversation.stream_text_deltas(initial_backend, request))

        self.assertEqual(deltas, ["second account response"])
        self.assertEqual(created_tokens, ["first-token", "second-token"])
        self.assertEqual(requested_attempts, [{"first-token"}])
        account_service.get_text_access_token.assert_called_once()
        account_service.remove_invalid_token.assert_not_called()
        account_service.mark_text_used.assert_called_once_with("second-token")


if __name__ == "__main__":
    unittest.main()
