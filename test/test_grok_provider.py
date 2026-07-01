from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any, Iterator, cast
from types import ModuleType

from test.optional_stubs import install_curl_cffi_stub, install_fastapi_stubs, install_pil_stub, install_pybase64_stub, install_pydantic_stub, install_starlette_stub, install_tiktoken_stub

install_curl_cffi_stub()
install_fastapi_stubs()
install_pil_stub()
install_pybase64_stub()
install_pydantic_stub()
install_starlette_stub()
install_tiktoken_stub()

import json
import sys
import types
import unittest
from unittest import mock

HTTPException = cast(Any, getattr(sys.modules["fastapi"], "HTTPException"))

from services.models import resolve_model
from services.network import flaresolverr
from services.protocol import openai_v1_chat_complete, openai_v1_image_edit, openai_v1_response
from services.protocol.conversation import ImageGenerationError, ImageOutput
from services.providers import registry as provider_registry
from services.providers.grok import accounts as grok_accounts
from services.providers.grok import client as grok
from services.providers.grok import images as grok_images
from services.providers.grok.models import GROK_MODEL_SPECS, is_supported_grok_app_chat_image_model


def _chat_chunks(chunks: object) -> list[Mapping[str, Any]]:
    return cast(list[Mapping[str, Any]], list(cast(Any, chunks)))


def _account_service_module(account_service: object) -> ModuleType:
    module = ModuleType("services.account_service")
    setattr(module, "account_service", account_service)
    return module


class GrokProviderTests(unittest.TestCase):
    def test_grok_model_specs_follow_upstream_mode_tier_table(self) -> None:
        expected = {
            "grok-4.20-0309-non-reasoning": ("fast", "basic", False, "chat"),
            "grok-4.20-0309": ("auto", "super", False, "chat"),
            "grok-4.20-0309-reasoning": ("expert", "super", False, "chat"),
            "grok-4.20-0309-non-reasoning-super": ("fast", "super", False, "chat"),
            "grok-4.20-0309-super": ("auto", "super", False, "chat"),
            "grok-4.20-0309-reasoning-super": ("expert", "super", False, "chat"),
            "grok-4.20-0309-non-reasoning-heavy": ("fast", "heavy", False, "chat"),
            "grok-4.20-0309-heavy": ("auto", "heavy", False, "chat"),
            "grok-4.20-0309-reasoning-heavy": ("expert", "heavy", False, "chat"),
            "grok-4.20-multi-agent-0309": ("heavy", "heavy", False, "chat"),
            "grok-4.20-fast": ("fast", "basic", True, "chat"),
            "grok-4.20-auto": ("auto", "super", True, "chat"),
            "grok-4.20-expert": ("expert", "super", True, "chat"),
            "grok-4.20-heavy": ("heavy", "heavy", True, "chat"),
            "grok-4.3-beta": ("grok-420-computer-use-sa", "super", False, "chat"),
            "grok-imagine-image-lite": ("fast", "basic", False, "image"),
            "grok-imagine-image": ("auto", "super", False, "image"),
            "grok-imagine-image-pro": ("auto", "super", False, "image"),
            "grok-imagine-image-edit": ("auto", "super", False, "image_edit"),
            "grok-imagine-video": ("auto", "super", False, "video"),
        }

        specs = {spec.id: spec for spec in GROK_MODEL_SPECS}

        for model_id, (mode_id, model_tier, prefer_best, capability) in expected.items():
            spec = specs[model_id]
            self.assertEqual(spec.mode_id, mode_id, model_id)
            self.assertEqual(spec.model_tier, model_tier, model_id)
            self.assertEqual(spec.prefer_best, prefer_best, model_id)
            self.assertEqual(spec.capability, capability, model_id)

    def test_grok_app_chat_model_and_image_support_metadata(self) -> None:
        self.assertFalse(openai_v1_chat_complete.is_grok_app_chat_model(resolve_model("grok-4.20-reasoning")))
        self.assertTrue(openai_v1_chat_complete.is_grok_app_chat_model(resolve_model("grok-4.20-fast")))
        self.assertTrue(openai_v1_chat_complete.is_grok_app_chat_model(resolve_model("grok-imagine-image-lite")))
        self.assertTrue(is_supported_grok_app_chat_image_model("grok-imagine-image-lite"))
        self.assertTrue(is_supported_grok_app_chat_image_model("grok-imagine-image-pro"))
        self.assertFalse(is_supported_grok_app_chat_image_model("grok-imagine-image-edit"))
        self.assertFalse(is_supported_grok_app_chat_image_model("grok-imagine-video"))

    def test_build_console_payload_converts_chat_messages(self) -> None:
        spec = resolve_model("grok-4.3")
        payload = grok.build_console_payload(
            spec,
            {"temperature": 0.2},
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": "Hi"},
            ],
        )

        self.assertEqual(payload["model"], "grok-4.3")
        self.assertEqual(payload["instructions"], "Be concise.")
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(payload["input"][0]["role"], "user")
        self.assertEqual(payload["input"][0]["content"], [{"type": "input_text", "text": "Hello"}])
        self.assertEqual(payload["input"][1]["content"], [{"type": "output_text", "text": "Hi"}])

    def test_build_console_payload_defaults_web_search_tool(self) -> None:
        spec = resolve_model("grok-4.3")
        payload = grok.build_console_payload(
            spec,
            {},
            [{"role": "user", "content": "Search the web."}],
        )

        self.assertEqual(payload["tools"], [{"type": "web_search"}])

    def test_build_console_payload_preserves_supported_search_tools(self) -> None:
        spec = resolve_model("grok-4.3")
        web_search = {"type": "web_search", "allowed_websites": ["example.com"]}
        x_search = {"type": "x_search", "post_favorite_count": 10}
        payload = grok.build_console_payload(
            spec,
            {"tools": [web_search, {"type": "image_generation"}, x_search]},
            [{"role": "user", "content": "Search the web."}],
        )

        self.assertEqual(payload["tools"], [web_search, x_search])

    def test_build_console_payload_appends_web_search_to_x_search_only(self) -> None:
        spec = resolve_model("grok-4.3")
        x_search = {"type": "x_search", "post_favorite_count": 10}
        payload = grok.build_console_payload(
            spec,
            {"tools": [{"type": "image_generation"}, x_search]},
            [{"role": "user", "content": "Search X and the web."}],
        )

        self.assertEqual(payload["tools"], [x_search, {"type": "web_search"}])

    def test_build_console_payload_preserves_response_tool_controls(self) -> None:
        spec = resolve_model("grok-4.3")
        payload = grok.build_console_payload(
            spec,
            {
                "tools": [{"type": "web_search"}],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            },
            [{"role": "user", "content": "Search the web"}],
        )

        self.assertEqual(payload["tools"], [{"type": "web_search"}])
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertTrue(payload["parallel_tool_calls"])

    def test_extract_console_text_from_common_shapes(self) -> None:
        self.assertEqual(grok.extract_console_text({"output_text": "direct"}), "direct")
        self.assertEqual(
            grok.extract_console_text({"output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}]}),
            "hello",
        )
        self.assertEqual(
            grok.extract_console_text({"output": [{"type": "output_text", "text": "hello"}, {"type": "text", "text": " world"}]}),
            "hello world",
        )

    def test_extract_console_completion_splits_chinese_visible_thinking(self) -> None:
        response = grok.extract_console_completion({
            "output_text": "**思考摘要**：先判断问题。\n\n继续分析。\n\n**答案**：最终回答。",
            "reasoning": {"effort": "high"},
            "usage": {"output_tokens_details": {"reasoning_tokens": 3}},
        })

        self.assertEqual(response.content, "最终回答。")
        self.assertEqual(response.reasoning_content, "先判断问题。\n\n继续分析。")
        self.assertEqual(response.raw_reasoning, {"effort": "high"})
        self.assertEqual(response.raw_usage, {"output_tokens_details": {"reasoning_tokens": 3}})

    def test_extract_console_completion_strips_bold_answer_marker_with_inner_colon(self) -> None:
        response = grok.extract_console_completion({
            "output_text": "**思考摘要**：先判断问题。\n\n**答案：**  \n8+8等于16。",
        })

        self.assertEqual(response.content, "8+8等于16。")
        self.assertEqual(response.reasoning_content, "先判断问题。")
        self.assertFalse(response.content.startswith("**"))

    def test_extract_console_completion_splits_english_visible_thinking(self) -> None:
        response = grok.extract_console_completion({
            "output_text": "**Thinking summary**: inspect inputs\nvalidate route\n\n**Answer**: use console",
        })

        self.assertEqual(response.content, "use console")
        self.assertEqual(response.reasoning_content, "inspect inputs\nvalidate route")

    def test_extract_console_completion_keeps_plain_content_unchanged(self) -> None:
        response = grok.extract_console_completion({"output_text": "plain answer"})

        self.assertEqual(response.content, "plain answer")
        self.assertEqual(response.reasoning_content, "")

    def test_extract_console_stream_delta_from_output_text_delta(self) -> None:
        delta = grok.extract_console_stream_delta({"type": "response.output_text.delta", "delta": "hello"})

        self.assertEqual(delta.content, "hello")
        self.assertEqual(delta.reasoning_content, "")

    def test_extract_console_stream_delta_from_reasoning_delta(self) -> None:
        delta = grok.extract_console_stream_delta({"type": "response.reasoning_summary_text.delta", "delta": "think"})

        self.assertEqual(delta.content, "")
        self.assertEqual(delta.reasoning_content, "think")

    def test_extract_console_stream_delta_ignores_completed_snapshot(self) -> None:
        delta = grok.extract_console_stream_delta({"type": "response.completed", "output_text": "complete text"})

        self.assertEqual(delta.content, "")
        self.assertEqual(delta.reasoning_content, "")

    def test_grok_account_normalizes_aliases_and_browser_metadata(self) -> None:
        account = grok_accounts.normalize_account({
            "access_token": "token-value",
            "tier": "SuperGrok",
            "status": "rate_limited",
            "capabilities": "Text Chat; image-generation; image-edit",
            "cfCookies": " CF_BM = one ; cf_clearance = stale ; CF_BM = two ",
            "cfClearance": " clearance-value ",
            "userAgent": " Account UA ",
            "statsigId": " statsig-account ",
            "sec-ch-ua": " sec ua ",
            "sec-ch-ua-mobile": " ?0 ",
            "sec-ch-ua-platform": " \"Linux\" ",
        })

        self.assertEqual(account["tier"], "super")
        self.assertEqual(account["status"], "限流")
        self.assertEqual(account["capabilities"], ["chat", "image", "image_edit"])
        self.assertEqual(account["cf_cookies"], "cf_bm=two; cf_clearance=stale")
        self.assertEqual(account["cf_clearance"], "clearance-value")
        self.assertEqual(account["user_agent"], "Account UA")
        self.assertEqual(account["statsig_id"], "statsig-account")
        self.assertEqual(account["sec_ch_ua"], "sec ua")
        self.assertEqual(account["sec_ch_ua_mobile"], "?0")
        self.assertEqual(account["sec_ch_ua_platform"], "\"Linux\"")

    def test_grok_account_normalizes_new_error_status_aliases(self) -> None:
        self.assertEqual(grok_accounts.normalize_status("unauthenticated"), "异常")
        self.assertEqual(grok_accounts.normalize_status("token_expired"), "异常")
        self.assertEqual(grok_accounts.normalize_status("rate_limit_exceeded"), "限流")

    def test_grok_account_normalizes_cookie_token_and_sso_fields(self) -> None:
        self.assertEqual(grok_accounts.normalize_access_token({"access_token": " SSO = cookie-token "}), "cookie-token")
        self.assertEqual(grok_accounts.normalize_access_token({"access_token": " SSO = cookie-token ; other=value "}), "SSO = cookie-token ; other=value")
        self.assertEqual(grok_accounts.normalize_access_token({"sso": " field-token "}), "field-token")
        self.assertEqual(grok_accounts.normalize_access_token({"access_token": "plain-token"}), "")
        self.assertEqual(grok_accounts.normalize_access_token({"sso-rw": " rw-token "}), "")

    def test_app_chat_headers_use_grok_app_shape_with_plain_token(self) -> None:
        with (
            mock.patch.object(grok, "_grok_app_chat_profile", return_value=types.SimpleNamespace(
                user_agent="Test UA",
                cf_clearance="",
                cf_cookies="",
                sec_ch_ua="test sec ua",
                sec_ch_ua_mobile="?0",
                sec_ch_ua_platform='"Windows"',
                statsig_id="statsig-test",
            )),
            mock.patch.object(grok.uuid, "uuid4", return_value="request-id"),
        ):
            headers = grok.app_chat_headers("plain-token")

        self.assertEqual(headers["Accept"], "*/*")
        self.assertEqual(headers["Accept-Encoding"], "gzip, deflate, br, zstd")
        self.assertEqual(headers["Accept-Language"], "zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["Origin"], "https://grok.com")
        self.assertEqual(headers["Referer"], "https://grok.com/")
        self.assertEqual(headers["Priority"], "u=1, i")
        self.assertEqual(headers["Sec-Fetch-Dest"], "empty")
        self.assertEqual(headers["Sec-Fetch-Mode"], "cors")
        self.assertEqual(headers["Sec-Fetch-Site"], "same-origin")
        self.assertEqual(headers["Sec-Ch-Ua"], "test sec ua")
        self.assertEqual(headers["Sec-Ch-Ua-Mobile"], "?0")
        self.assertEqual(headers["Sec-Ch-Ua-Platform"], '"Windows"')
        self.assertEqual(headers["User-Agent"], "Test UA")
        self.assertEqual(headers["x-statsig-id"], "statsig-test")
        self.assertEqual(headers["x-xai-request-id"], "request-id")
        self.assertEqual(headers["Cookie"], "sso=plain-token; sso-rw=plain-token")
        self.assertNotIn("cf_clearance", headers["Cookie"])
        self.assertNotIn("Authorization", headers)

    def test_app_chat_headers_normalize_simple_sso_cookie_token(self) -> None:
        with mock.patch.object(grok, "_grok_app_chat_profile", return_value=types.SimpleNamespace(
            user_agent="Test UA",
            cf_clearance="",
            cf_cookies="",
            sec_ch_ua="",
            sec_ch_ua_mobile="",
            sec_ch_ua_platform="",
            statsig_id=grok.GROK_APP_CHAT_STATSIG_ID,
        )):
            headers = grok.app_chat_headers(" sso=plain-token ")

        self.assertEqual(headers["Cookie"], "sso=plain-token; sso-rw=plain-token")
        self.assertNotIn("cf_clearance", headers["Cookie"])
        self.assertNotIn("Authorization", headers)

    def test_app_chat_headers_append_optional_cloudflare_profile_cookies(self) -> None:
        with mock.patch.object(grok, "_grok_app_chat_profile", return_value=types.SimpleNamespace(
            user_agent="Test UA",
            cf_clearance="profile-clearance",
            cf_cookies="cf_bm=profile-bm",
            sec_ch_ua="",
            sec_ch_ua_mobile="",
            sec_ch_ua_platform="",
            statsig_id=grok.GROK_APP_CHAT_STATSIG_ID,
        )):
            headers = grok.app_chat_headers("plain-token")

        self.assertEqual(headers["Cookie"], "sso=plain-token; sso-rw=plain-token; cf_bm=profile-bm; cf_clearance=profile-clearance")
        self.assertNotIn("Authorization", headers)

    def test_app_chat_cookie_merges_cloudflare_cookies_and_replaces_clearance(self) -> None:
        cookie = grok._app_chat_cookie(
            " sso=stored ; sso-rw=old ; cf_clearance=stored-clearance ; other=value ",
            " profile-clearance ",
            " cf_bm=profile-bm ; cf_clearance=cf-cookie-clearance ",
        )

        self.assertEqual(cookie, "sso=stored; sso-rw=stored; cf_clearance=profile-clearance; other=value; cf_bm=profile-bm")

    def test_app_chat_cookie_merges_solver_cookies_without_overriding_sso(self) -> None:
        cookie = grok._app_chat_cookie(
            " sso=stored ; sso-rw=old ",
            " solved-clearance ",
            " sso=solver ; sso-rw=solver-rw ; x-challenge=challenge ; x-signature=signature ; cf_clearance=solver-clearance ",
        )

        self.assertEqual(cookie, "sso=stored; sso-rw=stored; x-challenge=challenge; x-signature=signature; cf_clearance=solved-clearance")

    def test_app_chat_headers_normalize_cookie_token_without_overriding_clearance(self) -> None:
        with mock.patch.object(grok, "_grok_app_chat_profile", return_value=types.SimpleNamespace(
            user_agent="Test UA",
            cf_clearance="profile-clearance",
            cf_cookies="",
            sec_ch_ua="",
            sec_ch_ua_mobile="",
            sec_ch_ua_platform="",
            statsig_id=grok.GROK_APP_CHAT_STATSIG_ID,
        )):
            headers = grok.app_chat_headers(" sso=stored ; cf_clearance=stored-clearance ")

        self.assertEqual(headers["Cookie"], "sso=stored; sso-rw=stored; cf_clearance=profile-clearance")
        self.assertNotIn("Authorization", headers)

    def test_build_app_chat_payload_enables_web_search_for_text(self) -> None:
        spec = resolve_model("grok-4.20-heavy")
        payload = grok.build_app_chat_payload(
            spec,
            {},
            [{"role": "user", "content": "Search the web"}],
        )

        self.assertFalse(payload["disableSearch"])
        self.assertEqual(payload["toolOverrides"], {
            "imageGen": False,
            "webSearch": True,
            "xSearch": False,
            "xMediaSearch": False,
            "trendsSearch": False,
            "xPostAnalyze": False,
        })

    def test_build_app_chat_payload_uses_mode_tier_and_image_flags(self) -> None:
        spec = resolve_model("grok-4.20-heavy")
        payload = grok.build_app_chat_payload(
            spec,
            {"n": 2},
            [{"role": "user", "content": "Draw a cat"}],
            image_generation=True,
        )

        self.assertEqual(payload["message"], "Draw a cat")
        self.assertEqual(payload["modeId"], "heavy")
        self.assertEqual(payload["modelTier"], "heavy")
        self.assertTrue(payload["preferBest"])
        self.assertEqual(payload["collectionIds"], [])
        self.assertEqual(payload["connectors"], [])
        self.assertEqual(payload["deviceEnvInfo"], {
            "darkModeEnabled": False,
            "devicePixelRatio": 2,
            "screenHeight": 1329,
            "screenWidth": 2056,
            "viewportHeight": 1083,
            "viewportWidth": 2056,
        })
        self.assertTrue(payload["disableMemory"])
        self.assertFalse(payload["disableSearch"])
        self.assertFalse(payload["disableSelfHarmShortCircuit"])
        self.assertFalse(payload["disableTextFollowUps"])
        self.assertTrue(payload["enableImageGeneration"])
        self.assertTrue(payload["enableImageStreaming"])
        self.assertTrue(payload["enableSideBySide"])
        self.assertEqual(payload["fileAttachments"], [])
        self.assertFalse(payload["forceConcise"])
        self.assertFalse(payload["forceSideBySide"])
        self.assertEqual(payload["imageAttachments"], [])
        self.assertEqual(payload["imageGenerationCount"], 2)
        self.assertFalse(payload["isAsyncChat"])
        self.assertEqual(payload["responseMetadata"], {})
        self.assertFalse(payload["returnImageBytes"])
        self.assertFalse(payload["returnRawGrokInXaiRequest"])
        self.assertFalse(payload["searchAllConnectors"])
        self.assertTrue(payload["sendFinalMetadata"])
        self.assertTrue(payload["temporary"])
        self.assertEqual(payload["toolOverrides"], {
            "imageGen": False,
            "webSearch": False,
            "xSearch": False,
            "xMediaSearch": False,
            "trendsSearch": False,
            "xPostAnalyze": False,
        })

    def test_app_chat_reasoning_and_text_extraction(self) -> None:
        events = grok.app_chat_line_events([
            b'event: message',
            b'data: {"result":{"response":{"token":"plan ","isThinking":true}}}',
            b'',
            json.dumps({"data": {"result": {"response": {"token": "answer", "messageTag": "final"}}}}),
            'data: {"result":{"response":{"finalMetadata":{}}}}',
            '',
        ])

        response = grok.collect_app_chat_response(events)

        self.assertEqual(response["content"], "answer")
        self.assertEqual(response["reasoning_content"], "plan ")
        self.assertEqual(response["search_sources"], [])

    def test_app_chat_line_events_combines_multiline_data_event(self) -> None:
        events = list(grok.app_chat_line_events([
            'data: {"result":',
            'data: {"response":{"token":"split"}}}',
            '',
        ]))

        self.assertEqual(events, [{"result": {"response": {"token": "split"}}}])

    def test_app_chat_line_events_splits_embedded_data_frames_and_skips_bad_json(self) -> None:
        events = list(grok.app_chat_line_events([
            b'data: {"token":"first"}\n\ndata: not-json\n\ndata: {"token":"second","messageTag":"reasoning"}',
            b'',
        ]))

        self.assertEqual(events, [{"token": "first"}, {"token": "second", "messageTag": "reasoning"}])

    def test_collect_app_chat_response_handles_root_tokens_text_alias_reasoning_and_usage(self) -> None:
        events = [
            {"token": "plan ", "messageTag": "reasoning"},
            {"response": {"text": "answer"}},
            {"usage": {"total_tokens": 9}},
            {"finalMetadata": {"usage": {"total_tokens": 9}}},
        ]

        response = grok.collect_app_chat_response(events)

        self.assertEqual(response["content"], "answer")
        self.assertEqual(response["reasoning_content"], "plan ")
        self.assertEqual(response["search_sources"], [])

    def test_collect_app_chat_response_accumulates_final_tag_tokens(self) -> None:
        events = [
            {"result": {"response": {"token": "Hello", "messageTag": "final"}}},
            {"result": {"response": {"token": " world", "messageTag": "final"}}},
            {"result": {"response": {"isSoftStop": True}}},
        ]

        response = grok.collect_app_chat_response(events)

        self.assertEqual(response["content"], "Hello world")
        self.assertEqual(response["reasoning_content"], "")
        self.assertEqual(response["search_sources"], [])

    def test_collect_app_chat_response_extracts_search_sources(self) -> None:
        events = [
            {
                "result": {
                    "response": {
                        "token": "answer",
                        "webSearchResults": {
                            "results": [
                                {"url": "https://example.com/a", "title": "Example A"},
                                {"url": "https://example.com/a", "title": "Duplicate"},
                            ]
                        },
                    }
                }
            },
            {
                "result": {
                    "response": {
                        "finalMetadata": {
                            "sources": [
                                {"link": "https://example.com/b", "name": "Example B"},
                            ],
                            "xResults": [
                                {"url": "https://x.com/alice/status/123", "text": "  Hello   from X post with normalized text  "},
                            ],
                        },
                    }
                }
            },
        ]

        response = grok.collect_app_chat_response(events)

        self.assertEqual(response["content"], "answer")
        self.assertEqual(response["search_sources"], [
            {"url": "https://example.com/a", "title": "Example A", "type": "web"},
            {"url": "https://example.com/b", "title": "Example B", "type": "web"},
            {"url": "https://x.com/alice/status/123", "title": "Hello from X post with normalized text", "type": "x_post"},
        ])

    def test_format_search_sources_suffix_escapes_markdown_link_text(self) -> None:
        suffix = grok.format_search_sources_suffix([
            {"url": "https://example.com/a", "title": r"Look [here] \ now", "type": "web"},
        ])

        self.assertIn(r"1. [Look \[here\] \\ now](https://example.com/a)", suffix)

    def test_format_search_sources_suffix_filters_urls_that_break_markdown_links(self) -> None:
        suffix = grok.format_search_sources_suffix([
            {"url": "https://example.com/good", "title": "Good", "type": "web"},
            {"url": "https://example.com/bad)tail", "title": "Bad paren", "type": "web"},
            {"url": "https://example.com/bad path", "title": "Bad space", "type": "web"},
            {"url": "https://example.com/bad\npath", "title": "Bad control", "type": "web"},
        ])

        self.assertIn("1. [Good](https://example.com/good)", suffix)
        self.assertNotIn("Bad paren", suffix)
        self.assertNotIn("Bad space", suffix)
        self.assertNotIn("Bad control", suffix)

    def test_strip_search_sources_from_assistant_history(self) -> None:
        suffix = grok.format_search_sources_suffix([
            {"url": "https://example.com/a", "title": "Example A", "type": "web"},
        ])
        messages = [
            {"role": "assistant", "content": f"Answer{suffix}"},
            {"role": "user", "content": "next"},
        ]

        stripped = grok.strip_search_sources_from_messages(messages)

        self.assertEqual(stripped[0]["content"], "Answer")
        self.assertEqual(stripped[1]["content"], "next")

    def test_message_tag_final_is_not_app_chat_final_event(self) -> None:
        self.assertFalse(grok.is_app_chat_final_event({"result": {"response": {"messageTag": "final"}}}))
        self.assertTrue(grok.is_app_chat_final_event({"result": {"response": {"finalMetadata": {}}}}))
        self.assertTrue(grok.is_app_chat_final_event({"result": {"response": {"isSoftStop": True}}}))

    def test_extract_app_chat_image_url_from_final_chunk(self) -> None:
        event = {
            "result": {
                "response": {
                    "cardAttachment": {
                        "jsonData": {
                            "image_chunk": {
                                "progress": 100,
                                "imageUrl": "generated/cat.png",
                            }
                        }
                    }
                }
            }
        }

        self.assertEqual(grok.extract_app_chat_image_url(event), "https://assets.grok.com/generated/cat.png")

    def test_extract_app_chat_image_url_from_json_string_final_chunk(self) -> None:
        event = {
            "result": {
                "response": {
                    "cardAttachment": {
                        "jsonData": json.dumps({
                            "image_chunk": {
                                "progress": 100,
                                "imageUrl": "/generated/dog.png",
                            }
                        })
                    }
                }
            }
        }

        self.assertEqual(grok.extract_app_chat_image_url(event), "https://assets.grok.com/generated/dog.png")

    def test_extract_app_chat_image_url_accepts_final_url_aliases_and_snake_case_asset(self) -> None:
        event = {"image": {"progress": 100, "finalUrl": "generated/final.png"}}
        asset_event = {"media": {"progress": 100, "asset_id": "asset-1", "user_id": "user-1"}}

        self.assertEqual(grok.extract_app_chat_image_url(event), "https://assets.grok.com/generated/final.png")
        self.assertEqual(grok.extract_app_chat_image_url(asset_event), "https://assets.grok.com/users/user-1/asset-1/content")

    def test_extract_app_chat_image_url_from_card_media_and_metadata(self) -> None:
        event = {
            "result": {
                "response": {
                    "cardAttachment": {
                        "jsonData": json.dumps({
                            "media": {
                                "progress": 100,
                                "mediaUrl": "https://assets.grok.com/generated/card.png",
                            }
                        })
                    },
                    "finalMetadata": {
                        "image_chunk": {
                            "progress": 100,
                            "assetUrl": "generated/final.png",
                        }
                    },
                }
            }
        }

        self.assertEqual(grok.extract_app_chat_image_url(event), "https://assets.grok.com/generated/card.png")

    def test_app_chat_moderation_message_detects_blocked_image_chunks(self) -> None:
        event = {
            "result": {
                "response": {
                    "cardAttachment": {
                        "jsonData": {
                            "image_chunk": {
                                "progress": 100,
                                "blocked": True,
                                "reason": "Blocked by policy",
                                "imageUrl": "generated/blocked.png",
                            }
                        }
                    }
                }
            }
        }

        self.assertEqual(grok.extract_app_chat_image_url(event), "")
        self.assertEqual(grok.app_chat_moderation_message(event), "Blocked by policy")

    def test_streaming_grok_chat_completion_returns_openai_chunks(self) -> None:
        body = {
            "model": "grok-4.20-multi-agent",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        events = [
            {"type": "response.output_text.delta", "delta": "Hi"},
            {"type": "response.output_text.delta", "delta": " there"},
            {"type": "response.completed"},
        ]
        with (
            mock.patch.object(grok, "console_chat_completion_events", return_value=iter(events)) as patched_stream,
            mock.patch.object(grok, "console_chat_completion") as patched_blocking,
        ):
            chunks = _chat_chunks(openai_v1_chat_complete.handle(body))

        patched_stream.assert_called_once()
        patched_blocking.assert_not_called()
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["object"], "chat.completion.chunk")
        self.assertEqual(chunks[0]["model"], "grok-4.20-multi-agent")
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": "Hi"})
        self.assertIsNone(chunks[0]["choices"][0]["finish_reason"])
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": " there"})
        self.assertEqual(chunks[2]["choices"][0]["delta"], {})
        self.assertEqual(chunks[2]["choices"][0]["finish_reason"], "stop")

    def test_console_grok_reasoning_model_uses_console_path(self) -> None:
        spec = resolve_model("grok-4.20-reasoning")

        self.assertFalse(openai_v1_chat_complete.is_grok_app_chat_model(spec))

    def test_streaming_grok_console_completion_emits_reasoning_content(self) -> None:
        body = {
            "model": "grok-4.20-reasoning",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        events = [
            {"type": "response.reasoning_summary_text.delta", "delta": "think"},
            {"type": "response.output_text.delta", "delta": "Hi"},
        ]
        with (
            mock.patch.object(grok, "console_chat_completion_events", return_value=iter(events)) as patched_console,
            mock.patch.object(grok, "app_chat_completion_events") as patched_app_chat,
            mock.patch.object(grok, "console_chat_completion") as patched_blocking,
        ):
            chunks = _chat_chunks(openai_v1_chat_complete.handle(body))

        patched_console.assert_called_once()
        patched_app_chat.assert_not_called()
        patched_blocking.assert_not_called()
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant", "reasoning_content": "think"})
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "Hi"})
        self.assertEqual(chunks[2]["choices"][0]["finish_reason"], "stop")

    def test_streaming_grok_console_completion_includes_usage_when_requested(self) -> None:
        body = {
            "model": "grok-4.20-reasoning",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        events = [
            {"type": "response.reasoning_summary_text.delta", "delta": "think"},
            {"type": "response.output_text.delta", "delta": "Hi"},
        ]
        with (
            mock.patch.object(grok, "console_chat_completion_events", return_value=iter(events)) as patched_console,
            mock.patch.object(grok, "console_chat_completion") as patched_blocking,
            mock.patch.object(openai_v1_chat_complete, "count_message_tokens", return_value=11),
            mock.patch.object(openai_v1_chat_complete, "count_text_tokens", side_effect=[2, 3]),
        ):
            chunks = _chat_chunks(openai_v1_chat_complete.handle(body))

        patched_console.assert_called_once()
        patched_blocking.assert_not_called()

        self.assertIsNone(chunks[0]["usage"])
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"], {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16})

    def test_streaming_grok_app_chat_completion_emits_reasoning_content(self) -> None:
        body = {
            "model": "grok-4.20-heavy",
            "stream": True,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        events = [
            {"result": {"response": {"token": "think", "isThinking": True}}},
            {"result": {"response": {"token": "Hi", "messageTag": "final"}}},
        ]
        with mock.patch.object(grok, "app_chat_completion_events", return_value=iter(events)):
            chunks = _chat_chunks(openai_v1_chat_complete.handle(body))

        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant", "reasoning_content": "think"})
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "Hi"})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_streaming_grok_app_chat_completion_includes_usage_when_requested(self) -> None:
        body = {
            "model": "grok-4.20-heavy",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        events = [
            {"result": {"response": {"token": "think", "isThinking": True}}},
            {"result": {"response": {"token": "Hi"}}},
            {"result": {"response": {"token": " there", "messageTag": "final"}}},
        ]
        with (
            mock.patch.object(grok, "app_chat_completion_events", return_value=iter(events)),
            mock.patch.object(openai_v1_chat_complete, "count_message_tokens", return_value=7),
            mock.patch.object(openai_v1_chat_complete, "count_text_tokens", side_effect=[4, 2]),
        ):
            chunks = _chat_chunks(openai_v1_chat_complete.handle(body))

        self.assertIsNone(chunks[0]["usage"])
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"], {"prompt_tokens": 7, "completion_tokens": 6, "total_tokens": 13})

    def test_streaming_text_completion_includes_usage_when_requested(self) -> None:
        body = {
            "model": "auto",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with (
            mock.patch.object(openai_v1_chat_complete, "text_backend", return_value=object()),
            mock.patch.object(openai_v1_chat_complete, "ConversationRequest", return_value=object()),
            mock.patch.object(openai_v1_chat_complete, "stream_text_deltas", return_value=iter(["Hi", " there"])),
            mock.patch.object(openai_v1_chat_complete, "count_message_tokens", return_value=5),
            mock.patch.object(openai_v1_chat_complete, "count_text_tokens", return_value=8),
        ):
            chunks = _chat_chunks(openai_v1_chat_complete.handle(body))

        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant", "content": "Hi"})
        self.assertIsNone(chunks[0]["usage"])
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(chunks[-1]["usage"], {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13})

    def test_classify_app_chat_auth_json_marks_auth_failure_not_cloudflare(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())
        sys.modules["services.account_service"] = _account_service_module(account_service)
        response = mock.Mock()
        response.json.return_value = {"error": {"code": "unauthenticated", "message": "expired token"}}
        response.text = ""

        error = grok.classify_app_chat_upstream_error(401, "token-1", response)

        self.assertEqual(error.status_code, 401)
        self.assertEqual(error.code, "authentication_failed")
        account_service.update_account.assert_called_once_with("token-1", {"status": "异常"})

    def test_classify_app_chat_cloudflare_html_is_challenge_not_auth(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())
        sys.modules["services.account_service"] = _account_service_module(account_service)
        response = mock.Mock()
        response.json.side_effect = ValueError("not json")
        response.text = "<html><title>Just a moment...</title><script src='/cdn-cgi/challenge-platform/h/b'></script></html>"

        error = grok.classify_app_chat_upstream_error(403, "token-1", response)

        self.assertEqual(error.status_code, 502)
        self.assertEqual(error.code, "cloudflare_challenge")
        account_service.update_account.assert_not_called()

    def test_classify_app_chat_429_marks_rate_limit_not_auth_failure(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())
        sys.modules["services.account_service"] = _account_service_module(account_service)
        response = mock.Mock()
        response.json.return_value = {"error": {"code": "rate_limit_exceeded", "message": "Too many requests"}}
        response.text = ""

        error = grok.classify_app_chat_upstream_error(429, "token-1", response)

        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.code, "rate_limit_exceeded")
        account_service.update_account.assert_called_once_with("token-1", {"status": "限流"})

    def test_app_chat_completion_events_retries_second_token_after_rate_limit(self) -> None:
        calls: list[set[str]] = []

        def select_token(_spec: Any, excluded_tokens: set[str] | None = None) -> str:
            excluded = set(excluded_tokens or set())
            calls.append(excluded)
            return "token-2" if "token-1" in excluded else "token-1"

        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(side_effect=select_token),
            get_account=mock.Mock(side_effect=lambda token: {"access_token": token}),
            mark_text_used=mock.Mock(),
            refresh_accounts=mock.Mock(),
        )
        sys.modules["services.account_service"] = _account_service_module(account_service)
        spec = resolve_model("grok-4.20-heavy")

        def client_factory(access_token: str, _account: dict[str, Any] | None = None) -> mock.Mock:
            client = mock.Mock()
            client.__enter__ = mock.Mock(return_value=client)
            client.__exit__ = mock.Mock(return_value=None)
            if access_token == "token-1":
                client.stream_events.side_effect = grok.GrokConsoleError("limited", 429, 429, "rate_limit_exceeded")
            else:
                client.stream_events.return_value = iter([{"result": {"response": {"token": "ok", "isSoftStop": True}}}])
            return client

        with mock.patch.object(grok, "GrokAppChatClient", side_effect=client_factory):
            events = list(grok.app_chat_completion_events({"messages": [{"role": "user", "content": "Hello"}]}, spec, [{"role": "user", "content": "Hello"}]))

        self.assertEqual(events, [{"result": {"response": {"token": "ok", "isSoftStop": True}}}])
        self.assertEqual(calls, [set(), {"token-1"}])
        account_service.mark_text_used.assert_called_once_with("token-2")

    def test_app_chat_completion_events_does_not_retry_after_partial_output(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(return_value="token-1"),
            get_account=mock.Mock(return_value={"access_token": "token-1"}),
            mark_text_used=mock.Mock(),
            refresh_accounts=mock.Mock(),
        )
        sys.modules["services.account_service"] = _account_service_module(account_service)
        spec = resolve_model("grok-4.20-heavy")

        def partial_then_limited() -> Iterator[dict[str, Any]]:
            yield {"result": {"response": {"token": "partial"}}}
            raise grok.GrokConsoleError("limited", 429, 429, "rate_limit_exceeded")

        client = mock.Mock()
        client.__enter__ = mock.Mock(return_value=client)
        client.__exit__ = mock.Mock(return_value=None)
        client.stream_events.return_value = partial_then_limited()

        with mock.patch.object(grok, "GrokAppChatClient", return_value=client):
            events = grok.app_chat_completion_events({"messages": [{"role": "user", "content": "Hello"}]}, spec, [{"role": "user", "content": "Hello"}])
            self.assertEqual(next(events), {"result": {"response": {"token": "partial"}}})
            with self.assertRaises(HTTPException) as context:
                next(events)

        self.assertEqual(context.exception.status_code, 429)
        self.assertEqual(account_service.get_grok_app_chat_access_token.call_count, 1)
        account_service.mark_text_used.assert_not_called()

    def test_app_chat_completion_does_not_mix_retry_after_partial_output(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(return_value="token-1"),
            get_account=mock.Mock(return_value={"access_token": "token-1"}),
            mark_text_used=mock.Mock(),
            refresh_accounts=mock.Mock(),
        )
        sys.modules["services.account_service"] = _account_service_module(account_service)
        spec = resolve_model("grok-4.20-heavy")

        def partial_then_transient() -> Iterator[dict[str, Any]]:
            yield {"result": {"response": {"token": "partial"}}}
            raise grok.GrokConsoleError("temporary", 502, 502, "upstream_transient")

        client = mock.Mock()
        client.__enter__ = mock.Mock(return_value=client)
        client.__exit__ = mock.Mock(return_value=None)
        client.stream_events.return_value = partial_then_transient()

        with mock.patch.object(grok, "GrokAppChatClient", return_value=client):
            with self.assertRaises(HTTPException) as context:
                grok.app_chat_completion({"messages": [{"role": "user", "content": "Hello"}]}, spec, [{"role": "user", "content": "Hello"}])

        self.assertEqual(context.exception.status_code, 502)
        self.assertEqual(account_service.get_grok_app_chat_access_token.call_count, 1)
        account_service.mark_text_used.assert_not_called()

    def test_app_chat_completion_events_exhausted_retry_returns_last_error(self) -> None:
        excluded_calls: list[set[str]] = []
        tokens = iter(["token-1", "token-2", ""])

        def next_token(*args: object, **kwargs: object) -> str:
            excluded_tokens = kwargs.get("excluded_tokens")
            excluded_calls.append(set(cast(set[str], excluded_tokens)) if isinstance(excluded_tokens, set) else set())
            return next(tokens)

        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(side_effect=next_token),
            get_account=mock.Mock(side_effect=lambda token: {"access_token": token}),
            mark_text_used=mock.Mock(),
            refresh_accounts=mock.Mock(),
        )
        sys.modules["services.account_service"] = _account_service_module(account_service)
        client = mock.Mock()
        client.__enter__ = mock.Mock(return_value=client)
        client.__exit__ = mock.Mock(return_value=None)
        client.stream_events.side_effect = grok.GrokConsoleError("transient", 502, 503, "upstream_transient")
        spec = resolve_model("grok-4.20-heavy")

        with mock.patch.object(grok, "GrokAppChatClient", return_value=client):
            with self.assertRaises(HTTPException) as context:
                list(grok.app_chat_completion_events({"messages": [{"role": "user", "content": "Hello"}]}, spec, [{"role": "user", "content": "Hello"}]))

        self.assertEqual(context.exception.status_code, 502)
        self.assertEqual(context.exception.detail["code"], "upstream_transient")
        self.assertEqual(excluded_calls[0], set())
        self.assertEqual(excluded_calls[1], {"token-1"})
        self.assertEqual(excluded_calls[2], {"token-1", "token-2"})
        account_service.mark_text_used.assert_not_called()

    def test_app_chat_image_outputs_retries_second_token_after_rate_limit(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(side_effect=["token-1", "token-2"]),
            get_account=mock.Mock(side_effect=lambda token: {"access_token": token}),
            mark_text_used=mock.Mock(),
        )
        sys.modules["services.account_service"] = _account_service_module(account_service)
        spec = resolve_model("grok-imagine-image")

        def client_factory(access_token: str, _account: dict[str, Any] | None = None) -> mock.Mock:
            client = mock.Mock()
            client.__enter__ = mock.Mock(return_value=client)
            client.__exit__ = mock.Mock(return_value=None)
            if access_token == "token-1":
                client.stream_events.side_effect = grok.GrokConsoleError("limited", 429, 429, "rate_limit_exceeded")
            else:
                client.stream_events.return_value = iter([{"result": {"response": {"streamingImageGenerationResponse": {"progress": 100, "imageUrl": "generated/second.png"}, "isSoftStop": True}}}])
            return client

        with mock.patch.object(grok, "GrokAppChatClient", side_effect=client_factory):
            outputs = list(grok.app_chat_image_outputs({"prompt": "Draw", "response_format": "url"}, spec, "Draw", 1))

        self.assertEqual(outputs[-1].data, [{"url": "https://assets.grok.com/generated/second.png", "revised_prompt": "Draw"}])
        self.assertEqual(account_service.get_grok_app_chat_access_token.call_args_list[1].kwargs["excluded_tokens"], {"token-1"})
        account_service.mark_text_used.assert_called_once_with("token-2")

    def test_non_streaming_grok_app_chat_completion_includes_reasoning_content(self) -> None:
        body = {
            "model": "grok-4.20-heavy",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with mock.patch.object(grok, "app_chat_completion", return_value={"content": "Hi", "reasoning_content": "think"}):
            response = cast(dict[str, Any], openai_v1_chat_complete.handle(body))

        message = response["choices"][0]["message"]
        self.assertEqual(message["content"], "Hi")
        self.assertEqual(message["reasoning_content"], "think")

    def test_console_chat_completion_uses_reserved_console_quota(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_console_access_token=mock.Mock(return_value="grok-token"),
            get_account=mock.Mock(return_value={}),
            mark_grok_console_used=mock.Mock(),
        )
        response_json = {
            "output": [{"content": [{"type": "output_text", "text": "Hi"}]}],
        }
        client = mock.Mock()
        client.__enter__ = mock.Mock(return_value=client)
        client.__exit__ = mock.Mock(return_value=None)
        client.create_response.return_value = response_json

        with (
            mock.patch.dict(sys.modules, {"services.account_service": types.SimpleNamespace(account_service=account_service)}),
            mock.patch.object(grok, "GrokConsoleClient", return_value=client),
        ):
            completion = grok.console_chat_completion({}, resolve_model("grok-4.3"), [{"role": "user", "content": "Hello"}])

        self.assertEqual(completion.content, "Hi")
        account_service.get_grok_console_access_token.assert_called_once_with()
        account_service.mark_grok_console_used.assert_not_called()

    def test_console_chat_completion_marks_failed_request_without_extra_quota_decrement(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_console_access_token=mock.Mock(return_value="grok-token"),
            get_account=mock.Mock(return_value={}),
            mark_grok_console_used=mock.Mock(),
        )
        client = mock.Mock()
        client.__enter__ = mock.Mock(return_value=client)
        client.__exit__ = mock.Mock(return_value=None)
        client.create_response.side_effect = grok.GrokConsoleError("upstream failed", 502)

        with (
            mock.patch.dict(sys.modules, {"services.account_service": types.SimpleNamespace(account_service=account_service)}),
            mock.patch.object(grok, "GrokConsoleClient", return_value=client),
        ):
            with self.assertRaises(grok.HTTPException):
                grok.console_chat_completion({}, resolve_model("grok-4.3"), [{"role": "user", "content": "Hello"}])

        account_service.get_grok_console_access_token.assert_called_once_with()
        account_service.mark_grok_console_used.assert_called_once_with("grok-token", success=False)

    def test_console_chat_completion_marks_empty_response_failed_without_extra_quota_decrement(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_console_access_token=mock.Mock(return_value="grok-token"),
            get_account=mock.Mock(return_value={}),
            mark_grok_console_used=mock.Mock(),
        )
        client = mock.Mock()
        client.__enter__ = mock.Mock(return_value=client)
        client.__exit__ = mock.Mock(return_value=None)
        client.create_response.return_value = {"output": []}

        with (
            mock.patch.dict(sys.modules, {"services.account_service": types.SimpleNamespace(account_service=account_service)}),
            mock.patch.object(grok, "GrokConsoleClient", return_value=client),
        ):
            with self.assertRaises(grok.HTTPException) as ctx:
                grok.console_chat_completion({}, resolve_model("grok-4.3"), [{"role": "user", "content": "Hello"}])

        self.assertEqual(ctx.exception.status_code, 502)
        account_service.get_grok_console_access_token.assert_called_once_with()
        client.create_response.assert_called_once()
        account_service.mark_grok_console_used.assert_called_once_with("grok-token", success=False)

    def test_console_chat_completion_validates_payload_before_reserving_quota(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_console_access_token=mock.Mock(return_value="grok-token"),
            get_account=mock.Mock(return_value={}),
            mark_grok_console_used=mock.Mock(),
        )

        with mock.patch.dict(sys.modules, {"services.account_service": types.SimpleNamespace(account_service=account_service)}):
            with self.assertRaises(grok.HTTPException):
                grok.console_chat_completion({}, resolve_model("grok-4.3"), [{"role": "user", "content": ""}])

        account_service.get_grok_console_access_token.assert_not_called()
        account_service.mark_grok_console_used.assert_not_called()

    def test_console_stream_uses_reserved_console_quota(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_console_access_token=mock.Mock(return_value="grok-token"),
            get_account=mock.Mock(return_value={}),
            mark_grok_console_used=mock.Mock(),
        )
        client = mock.Mock()
        client.__enter__ = mock.Mock(return_value=client)
        client.__exit__ = mock.Mock(return_value=None)
        client.stream_response.return_value = iter([{"type": "response.created"}, {"type": "response.completed"}])

        with (
            mock.patch.dict(sys.modules, {"services.account_service": types.SimpleNamespace(account_service=account_service)}),
            mock.patch.object(grok, "GrokConsoleClient", return_value=client),
        ):
            events = list(grok.console_chat_completion_events({}, resolve_model("grok-4.3"), [{"role": "user", "content": "Hello"}]))

        self.assertEqual([event["type"] for event in events], ["response.created", "response.completed"])
        account_service.get_grok_console_access_token.assert_called_once_with()
        account_service.mark_grok_console_used.assert_not_called()

    def test_non_streaming_grok_console_completion_includes_reasoning_content(self) -> None:
        body = {
            "model": "grok-4.20-reasoning",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        with mock.patch.object(
            grok,
            "console_chat_completion",
            return_value=grok.GrokConsoleCompletion(content="Hi", reasoning_content="think"),
        ) as patched_console, mock.patch.object(grok, "app_chat_completion") as patched_app_chat:
            response = cast(dict[str, Any], openai_v1_chat_complete.handle(body))

        patched_console.assert_called_once()
        patched_app_chat.assert_not_called()
        message = response["choices"][0]["message"]
        self.assertEqual(message["content"], "Hi")
        self.assertEqual(message["reasoning_content"], "think")

    def test_responses_grok_console_routes_to_console_completion(self) -> None:
        body = {
            "model": "grok-4.3",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "tools": [{"type": "web_search"}],
        }
        with mock.patch.object(
            grok,
            "console_chat_completion",
            return_value=grok.GrokConsoleCompletion(content="Hi from Grok"),
        ) as patched_console:
            response = cast(dict[str, Any], openai_v1_response.handle(body))

        patched_console.assert_called_once()
        self.assertEqual(patched_console.call_args.args[0]["tools"], [{"type": "web_search"}])
        self.assertEqual(response["object"], "response")
        self.assertEqual(response["status"], "completed")
        content = response["output"][0]["content"][0]
        self.assertEqual(content["type"], "output_text")
        self.assertEqual(content["text"], "Hi from Grok")

    def test_streaming_responses_grok_console_emits_response_events(self) -> None:
        body = {
            "model": "grok-4.3",
            "input": "Hello",
            "stream": True,
        }
        with mock.patch.object(
            grok,
            "console_chat_completion",
            return_value=grok.GrokConsoleCompletion(content="Hi"),
        ) as patched_console:
            events = cast(list[dict[str, Any]], list(openai_v1_response.handle(body)))

        patched_console.assert_called_once()
        event_types = [event.get("type") for event in events]
        self.assertEqual(event_types[0], "response.created")
        self.assertIn("response.output_text.delta", event_types)
        self.assertEqual(event_types[-1], "response.completed")

    def test_responses_unknown_non_grok_model_uses_text_backend(self) -> None:
        body = {
            "model": "custom-text-model",
            "input": "Hello",
        }
        with (
            mock.patch.object(openai_v1_response, "text_backend", return_value=object()),
            mock.patch.object(openai_v1_response, "ConversationRequest", lambda **kwargs: kwargs),
            mock.patch.object(openai_v1_response, "stream_text_deltas", return_value=iter(["generic"])) as patched_stream,
            mock.patch.object(grok, "console_chat_completion") as patched_console,
        ):
            response = cast(dict[str, Any], openai_v1_response.handle(body))

        patched_stream.assert_called_once()
        patched_console.assert_not_called()
        self.assertEqual(response["output"][0]["content"][0]["text"], "generic")

    def test_responses_extract_image_accepts_input_image_data_url_object(self) -> None:
        png_bytes = b"\x89PNG\r\n\x1a\n"
        data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"

        image = openai_v1_response.extract_response_image([
            {"type": "input_text", "text": "edit this"},
            {"type": "input_image", "image_url": {"url": data_url}},
        ])

        self.assertEqual(image, (png_bytes, "image/png"))

    def test_responses_extract_image_accepts_content_image_url_object(self) -> None:
        gif_bytes = b"GIF89a"
        data_url = f"data:image/gif;base64,{base64.b64encode(gif_bytes).decode('ascii')}"

        image = openai_v1_response.extract_response_image({
            "role": "user",
            "content": [
                {"type": "input_text", "text": "edit this"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        })

        self.assertEqual(image, (gif_bytes, "image/gif"))

    def test_responses_image_tool_preserves_request_dispatch_with_image_input(self) -> None:
        png_bytes = b"\x89PNG\r\n\x1a\n"
        data_url = f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"
        body = {
            "model": "gpt-image-2",
            "input": [
                {"type": "input_text", "text": "make it brighter"},
                {"type": "input_image", "image_url": {"url": data_url}},
            ],
            "tools": [{"type": "image_generation"}],
        }
        outputs = [ImageOutput(kind="result", model="gpt-image-2", index=1, total=1, data=[{"b64_json": "abc"}])]

        with mock.patch.object(openai_v1_response, "response_image_outputs", return_value=iter(outputs)) as patched_outputs:
            response = cast(dict[str, Any], openai_v1_response.handle(body))

        request = patched_outputs.call_args.args[1]
        self.assertEqual(request.prompt, "make it brighter")
        self.assertEqual(request.images, [base64.b64encode(png_bytes).decode("ascii")])
        self.assertEqual(response["output"][0]["type"], "image_generation_call")
        self.assertEqual(response["output"][0]["result"], "abc")

    def test_responses_image_tool_still_dispatches_without_image_input(self) -> None:
        body = {
            "model": "gpt-image-2",
            "input": [{"type": "input_text", "text": "draw a cat"}],
            "tools": [{"type": "image_generation"}],
        }
        outputs = [ImageOutput(kind="result", model="gpt-image-2", index=1, total=1, data=[{"b64_json": "abc"}])]

        with mock.patch.object(openai_v1_response, "response_image_outputs", return_value=iter(outputs)) as patched_outputs:
            response = cast(dict[str, Any], openai_v1_response.handle(body))

        request = patched_outputs.call_args.args[1]
        self.assertEqual(request.prompt, "draw a cat")
        self.assertIsNone(request.images)
        self.assertEqual(request.size, "1:1")
        self.assertEqual(response["output"][0]["result"], "abc")

    def test_responses_grok_app_chat_returns_explicit_error(self) -> None:
        body = {
            "model": "grok-4.20-heavy",
            "input": "Hello",
        }
        with self.assertRaises(HTTPException) as ctx:
            list(openai_v1_response.handle(body))

        self.assertEqual(getattr(ctx.exception, "status_code", None), 501)
        self.assertIn("Grok app-chat is not supported", str(getattr(ctx.exception, "detail", "")))

    def test_registry_image_generation_dispatches_by_provider(self) -> None:
        outputs = [ImageOutput(kind="result", model="gpt-image-2", index=1, total=1, data=[{"b64_json": "abc"}])]
        gpt_adapter = mock.Mock()
        grok_adapter = mock.Mock()
        gemini_adapter = mock.Mock()
        gpt_adapter.generation_outputs.return_value = iter(outputs)
        grok_adapter.generation_outputs.return_value = iter(outputs)
        gemini_adapter.generation_outputs.return_value = iter(outputs)

        def adapter_for(provider: object):
            return {
                "gpt": gpt_adapter,
                "grok": grok_adapter,
                "gemini": gemini_adapter,
            }[str(provider)]

        request = mock.Mock()
        with mock.patch.object(provider_registry, "image_adapter", side_effect=adapter_for):
            self.assertIs(provider_registry.image_generation_outputs(resolve_model("gpt-image-2"), request), gpt_adapter.generation_outputs.return_value)
            self.assertIs(provider_registry.image_generation_outputs(resolve_model("grok-imagine-image-lite"), request, body={"model": "grok-imagine-image-lite"}, prompt="cat", n=2), grok_adapter.generation_outputs.return_value)
            self.assertIs(provider_registry.image_generation_outputs(resolve_model("gemini-2.5-pro"), request), gemini_adapter.generation_outputs.return_value)

        gpt_adapter.generation_outputs.assert_called_once_with(request, resolve_model("gpt-image-2"))
        grok_adapter.generation_outputs.assert_called_once_with({"model": "grok-imagine-image-lite"}, resolve_model("grok-imagine-image-lite"), "cat", 2)
        gemini_adapter.generation_outputs.assert_called_once_with(request, resolve_model("gemini-2.5-pro"))

    def test_grok_image_lite_chat_routes_to_app_chat_image_outputs(self) -> None:
        body = {
            "model": "grok-imagine-image-lite",
            "messages": [{"role": "user", "content": "Draw a cat"}],
        }
        outputs = [ImageOutput(kind="result", model="grok-imagine-image-lite", index=1, total=1, data=[{"url": "https://assets.grok.com/cat.png"}])]
        result = {"created": 1, "data": [{"b64_json": "abc", "url": "https://assets.grok.com/cat.png"}]}
        with (
            mock.patch.object(grok_images, "generation_outputs", return_value=iter(outputs)) as patched,
            mock.patch.object(openai_v1_chat_complete, "collect_image_outputs", return_value=result),
        ):
            response = cast(dict[str, Any], openai_v1_chat_complete.handle(body))

        patched.assert_called_once()
        self.assertIn("data:image/png;base64,abc", response["choices"][0]["message"]["content"])

    def test_grok_image_edit_protocol_routes_to_app_chat_edit_outputs(self) -> None:
        body = {
            "model": "grok-imagine-image-edit",
            "prompt": "edit",
            "images": [(b"png", "input.png", "image/png")],
            "n": 1,
            "size": "1024x1024",
            "response_format": "url",
        }
        outputs = [ImageOutput(kind="result", model="grok-imagine-image-edit", index=1, total=1, data=[{"url": "https://assets.grok.com/edit.png"}])]
        with mock.patch.object(grok_images, "edit_outputs", return_value=iter(outputs)) as patched:
            response = cast(dict[str, Any], openai_v1_image_edit.handle(body))

        patched.assert_called_once()
        self.assertEqual(response["data"], [{"url": "https://assets.grok.com/edit.png"}])

    def test_grok_non_image_edit_model_still_unsupported_for_edits(self) -> None:
        body = {
            "model": "grok-imagine-image-lite",
            "prompt": "edit",
            "images": [(b"png", "input.png", "image/png")],
            "n": 1,
        }
        with self.assertRaises(ImageGenerationError) as context:
            openai_v1_image_edit.handle(body)

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.code, "unsupported_model")
        self.assertEqual(context.exception.param, "model")

    def test_unsupported_grok_image_model_raises_openai_error(self) -> None:
        spec = resolve_model("grok-imagine-image-edit")
        with self.assertRaises(ImageGenerationError) as context:
            list(grok.app_chat_image_outputs({"prompt": "Draw"}, spec, "Draw"))

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.code, "unsupported_model")
        self.assertEqual(context.exception.param, "model")

    def test_build_app_chat_payload_maps_image_size_to_aspect_config(self) -> None:
        spec = resolve_model("grok-imagine-image-pro")
        payload = grok.build_app_chat_payload(
            spec,
            {"n": 2, "size": "1792x1024"},
            [{"role": "user", "content": "Draw wide art"}],
            image_generation=True,
        )

        self.assertEqual(payload["imageGenerationCount"], 2)
        self.assertEqual(payload["modelConfigOverride"], {
            "modelMap": {
                "imageGenModel": "grok-imagine-image-pro",
                "imageGenModelConfig": {"aspectRatio": "3:2"},
            }
        })

    def test_build_app_chat_payload_accepts_ratio_size_without_affecting_text_chat(self) -> None:
        spec = resolve_model("grok-imagine-image")
        payload = grok.build_app_chat_payload(
            spec,
            {"size": "9:16"},
            [{"role": "user", "content": "Draw tall art"}],
            image_generation=True,
        )
        text_payload = grok.build_app_chat_payload(
            resolve_model("grok-4.20-heavy"),
            {"size": "1024x1792"},
            [{"role": "user", "content": "Hello"}],
        )

        self.assertEqual(payload["modelConfigOverride"]["modelMap"]["imageGenModelConfig"], {"aspectRatio": "9:16"})
        self.assertNotIn("modelConfigOverride", text_payload)

    def test_app_chat_image_outputs_dedupes_urls_truncates_to_n_and_keeps_progress(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(return_value="selected-token"),
            get_account=mock.Mock(return_value={"access_token": "selected-token"}),
            mark_text_used=mock.Mock(),
        )
        account_module = types.ModuleType("services.account_service")
        setattr(account_module, "account_service", account_service)
        sys.modules["services.account_service"] = account_module
        spec = resolve_model("grok-imagine-image")
        events = [
            {"result": {"response": {"token": "working"}}},
            {"result": {"response": {"streamingImageGenerationResponse": {"progress": 100, "imageUrl": "generated/one.png"}}}},
            {"result": {"response": {"streamingImageGenerationResponse": {"progress": 100, "imageUrl": "generated/one.png"}}}},
            {"result": {"response": {"streamingImageGenerationResponse": {"progress": 100, "imageUrl": "generated/two.png"}}}},
            {"result": {"response": {"streamingImageGenerationResponse": {"progress": 100, "imageUrl": "generated/three.png"}, "isSoftStop": True}}},
        ]

        with mock.patch.object(grok, "GrokAppChatClient") as client_class:
            client_class.return_value.__enter__.return_value.stream_events.return_value = iter(events)
            outputs = list(grok.app_chat_image_outputs(
                {"prompt": "Draw", "response_format": "url"},
                spec,
                "Draw",
                2,
            ))

        self.assertEqual(outputs[0].kind, "progress")
        self.assertEqual(outputs[0].text, "working")
        self.assertEqual(outputs[1].kind, "result")
        self.assertEqual(outputs[1].data, [
            {"url": "https://assets.grok.com/generated/one.png", "revised_prompt": "Draw"},
            {"url": "https://assets.grok.com/generated/two.png", "revised_prompt": "Draw"},
        ])
        account_service.mark_text_used.assert_called_once_with("selected-token")

    def test_app_chat_image_outputs_rejects_invalid_response_format(self) -> None:
        spec = resolve_model("grok-imagine-image")

        with self.assertRaises(ImageGenerationError) as context:
            list(grok.app_chat_image_outputs({"prompt": "Draw", "response_format": "xml"}, spec, "Draw", 1))

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.code, "invalid_response_format")
        self.assertEqual(context.exception.param, "response_format")

    def test_app_chat_image_outputs_defaults_to_b64_json_response_format(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(return_value="selected-token"),
            get_account=mock.Mock(return_value={"access_token": "selected-token"}),
            mark_text_used=mock.Mock(),
        )
        account_module = types.ModuleType("services.account_service")
        setattr(account_module, "account_service", account_service)
        sys.modules["services.account_service"] = account_module
        spec = resolve_model("grok-imagine-image-lite")
        events = [
            {"result": {"response": {"streamingImageGenerationResponse": {"progress": 100, "imageUrl": "generated/default.png"}, "isSoftStop": True}}},
        ]
        session = mock.Mock()
        session.get.return_value = types.SimpleNamespace(status_code=200, content=b"default")

        with (
            mock.patch.object(grok, "GrokAppChatClient") as client_class,
            mock.patch.object(grok, "create_session", return_value=session),
        ):
            client_class.return_value.__enter__.return_value.stream_events.return_value = iter(events)
            outputs = list(grok.app_chat_image_outputs({"prompt": "Draw"}, spec, "Draw", 1))

        self.assertEqual(outputs[-1].data, [{"b64_json": "ZGVmYXVsdA==", "revised_prompt": "Draw"}])

    def test_app_chat_image_outputs_respects_b64_json_response_format_for_grok_assets(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(return_value="selected-token"),
            get_account=mock.Mock(return_value={"access_token": "selected-token"}),
            mark_text_used=mock.Mock(),
        )
        account_module = types.ModuleType("services.account_service")
        setattr(account_module, "account_service", account_service)
        sys.modules["services.account_service"] = account_module
        spec = resolve_model("grok-imagine-image-lite")
        events = [
            {"result": {"response": {"streamingImageGenerationResponse": {"progress": 100, "imageUrl": "generated/cat.png"}, "isSoftStop": True}}},
        ]
        session = mock.Mock()
        session.get.return_value = types.SimpleNamespace(status_code=200, content=b"png")

        with (
            mock.patch.object(grok, "GrokAppChatClient") as client_class,
            mock.patch.object(grok, "create_session", return_value=session),
        ):
            client_class.return_value.__enter__.return_value.stream_events.return_value = iter(events)
            outputs = list(grok.app_chat_image_outputs(
                {"prompt": "Draw", "response_format": "b64_json"},
                spec,
                "Draw",
                1,
            ))

        session.get.assert_called_once_with("https://assets.grok.com/generated/cat.png", timeout=grok.GROK_IMAGE_DOWNLOAD_TIMEOUT)
        session.close.assert_called_once_with()
        self.assertEqual(outputs[0].data, [{"b64_json": "cG5n", "revised_prompt": "Draw"}])

    def test_grok_asset_base64_rejects_non_grok_asset_urls_without_download(self) -> None:
        session = mock.Mock()
        unsafe_urls = [
            "http://assets.grok.com/generated/cat.png",
            "https://evil.com/generated/cat.png",
            "https://assets.grok.com.evil.com/generated/cat.png",
            "https://assets.grok.com@evil.com/generated/cat.png",
            "//evil.com/generated/cat.png",
            "javascript:alert(1)",
            "bad\\path.png",
        ]

        with mock.patch.object(grok, "create_session", return_value=session):
            for url in unsafe_urls:
                with self.subTest(url=url), self.assertRaises(ImageGenerationError):
                    grok._grok_asset_base64(url)

        session.get.assert_not_called()

    def test_app_chat_image_edit_outputs_respects_base64_response_format(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(return_value="selected-token"),
            get_account=mock.Mock(return_value={"access_token": "selected-token"}),
            mark_text_used=mock.Mock(),
        )
        account_module = types.ModuleType("services.account_service")
        setattr(account_module, "account_service", account_service)
        sys.modules["services.account_service"] = account_module
        spec = resolve_model("grok-imagine-image-edit")
        session = mock.Mock()
        session.get.return_value = types.SimpleNamespace(status_code=200, content=b"edit")

        with (
            mock.patch.object(grok, "GrokAppChatClient") as client_class,
            mock.patch.object(grok, "create_session", return_value=session),
        ):
            client = client_class.return_value.__enter__.return_value
            client.upload_image_edit_reference.return_value = grok.GrokImageEditReference("file-1", "https://assets.grok.com/input.png")
            client.create_image_edit_parent_post.return_value = ("post-1", "edit")
            client.stream_image_edit_events.return_value = iter([
                {"result": {"response": {"streamingImageGenerationResponse": {"progress": 100, "imageUrl": "generated/edit.png"}, "isSoftStop": True}}},
            ])
            outputs = list(grok.app_chat_image_edit_outputs(
                {"prompt": "edit", "response_format": "base64"},
                spec,
                "edit",
                [(b"png", "input.png", "image/png")],
                1,
                "1024x1024",
            ))

        self.assertEqual(outputs[-1].data, [{"base64": "ZWRpdA==", "revised_prompt": "edit"}])

    def test_app_chat_image_outputs_errors_when_final_response_has_no_image(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(return_value="selected-token"),
            get_account=mock.Mock(return_value={"access_token": "selected-token"}),
            mark_text_used=mock.Mock(),
        )
        account_module = types.ModuleType("services.account_service")
        setattr(account_module, "account_service", account_service)
        sys.modules["services.account_service"] = account_module
        spec = resolve_model("grok-imagine-image")

        with mock.patch.object(grok, "GrokAppChatClient") as client_class:
            client_class.return_value.__enter__.return_value.stream_events.return_value = iter([{"result": {"response": {"isSoftStop": True}}}])
            with self.assertRaises(ImageGenerationError) as context:
                list(grok.app_chat_image_outputs({"prompt": "Draw"}, spec, "Draw", 1))

        self.assertEqual(context.exception.status_code, 502)
        self.assertEqual(context.exception.code, "image_generation_failed")
        self.assertIn("final response", str(context.exception))
        account_service.mark_text_used.assert_called_once_with("selected-token")

    def test_grok_image_edit_payload_uses_uploaded_references_and_parent_post(self) -> None:
        payload = grok.build_grok_image_edit_payload(
            "edit @file-1",
            ["https://assets.grok.com/input.png"],
            "post-1",
        )

        self.assertEqual(payload["modelName"], "imagine-image-edit")
        self.assertEqual(payload["message"], "edit @file-1")
        self.assertTrue(payload["enableImageGeneration"])
        self.assertTrue(payload["enableImageStreaming"])
        self.assertEqual(payload["imageGenerationCount"], 2)
        config = payload["responseMetadata"]["modelConfigOverride"]["modelMap"]["imageEditModelConfig"]
        self.assertEqual(config["imageReferences"], ["https://assets.grok.com/input.png"])
        self.assertEqual(config["parentPostId"], "post-1")

    def test_grok_image_edit_upload_payload_and_placeholder_replacement(self) -> None:
        calls = []
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client.access_token = "sso=token; x-userid=user-1"

        def fake_post(url, payload, *, context, referer=None):
            calls.append((url, payload, context, referer))
            return {"fileMetadataId": "file-1", "fileUri": "/users/user-1/file-1/content"}

        client._post_direct_json = fake_post
        reference = client.upload_image_edit_reference(b"png", "input.png", "image/png")

        self.assertEqual(calls[0][0], grok.APP_CHAT_UPLOAD_FILE_URL)
        self.assertEqual(calls[0][1], {"fileName": "input.png", "fileMimeType": "image/png", "content": "cG5n"})
        self.assertEqual(reference.content_url, "https://assets.grok.com/users/user-1/file-1/content")
        self.assertEqual(grok.replace_grok_image_placeholders("edit @IMAGE1 and @IMAGE2", [reference]), "edit @file-1 and @IMAGE2")

    def test_grok_image_edit_upload_accepts_final_url_variants(self) -> None:
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client.access_token = "sso=token"

        def fake_post(url, payload, *, context, referer=None):
            return {"fileID": "file-1", "contentURL": "generated/uploaded.png"}

        client._post_direct_json = fake_post
        reference = client.upload_image_edit_reference(b"png", "input.png", "image/png")

        self.assertEqual(reference.file_id, "file-1")
        self.assertEqual(reference.content_url, "https://assets.grok.com/generated/uploaded.png")

    def test_grok_image_edit_extracts_stream_and_model_response_urls(self) -> None:
        stream_event = {
            "result": {
                "response": {
                    "streamingImageGenerationResponse": {
                        "progress": 100,
                        "imageIndex": 1,
                        "imageUrl": "/generated/edit.png",
                    }
                }
            }
        }
        model_event = {
            "result": {
                "response": {
                    "modelResponse": {
                        "generatedImageUrls": ["https://assets.grok.com/generated/fallback.png"],
                        "fileAttachments": ["asset-1"],
                    }
                }
            }
        }
        generated_event = {
            "result": {
                "response": {
                    "modelResponse": {
                        "generatedImages": [{"contentUrl": "generated/object.png"}],
                    }
                }
            }
        }

        self.assertEqual(grok.extract_grok_image_edit_final_urls(stream_event), {1: "https://assets.grok.com/generated/edit.png"})
        self.assertEqual(
            grok.extract_grok_image_edit_final_urls(model_event, "user-1"),
            {0: "https://assets.grok.com/users/user-1/asset-1/content"},
        )
        self.assertEqual(grok.extract_grok_image_edit_final_urls(generated_event), {0: "https://assets.grok.com/generated/object.png"})

    def test_grok_image_edit_extracts_final_metadata_urls(self) -> None:
        event = {
            "result": {
                "response": {
                    "finalMetadata": {
                        "streamingImageGenerationResponse": {
                            "progress": 100,
                            "imageIndex": 0,
                            "assetUrl": "generated/final-edit.png",
                        }
                    }
                }
            }
        }

        self.assertEqual(grok.extract_grok_image_edit_final_urls(event), {0: "https://assets.grok.com/generated/final-edit.png"})

    def test_grok_image_edit_rejects_non_grok_asset_urls(self) -> None:
        event = {
            "result": {
                "response": {
                    "streamingImageGenerationResponse": {
                        "progress": 100,
                        "imageUrl": "https://example.com/generated/edit.png",
                    },
                    "modelResponse": {
                        "generatedImageUrls": ["http://assets.grok.com/insecure.png"],
                    },
                }
            }
        }

        self.assertEqual(grok.extract_grok_image_edit_final_urls(event), {})

    def test_grok_image_edit_outputs_errors_when_no_final_result(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(return_value="selected-token"),
            get_account=mock.Mock(return_value={"access_token": "selected-token"}),
            mark_text_used=mock.Mock(),
        )
        sys.modules["services.account_service"] = _account_service_module(account_service)
        spec = resolve_model("grok-imagine-image-edit")

        with mock.patch.object(grok, "GrokAppChatClient") as client_class:
            client = client_class.return_value.__enter__.return_value
            client.upload_image_edit_reference.return_value = grok.GrokImageEditReference("file-1", "https://assets.grok.com/input.png")
            client.create_image_edit_parent_post.return_value = ("post-1", "edit")
            client.stream_image_edit_events.return_value = iter([{"result": {"response": {"isSoftStop": True}}}])
            with self.assertRaises(ImageGenerationError) as context:
                list(grok.app_chat_image_edit_outputs(
                    {"prompt": "edit"},
                    spec,
                    "edit",
                    [(b"png", "input.png", "image/png")],
                    1,
                    "1024x1024",
                ))

        self.assertEqual(context.exception.status_code, 502)
        self.assertEqual(context.exception.code, "image_edit_failed")
        account_service.mark_text_used.assert_called_once_with("selected-token")

    def test_grok_image_edit_outputs_content_policy_error_when_blocked(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(return_value="selected-token"),
            get_account=mock.Mock(return_value={"access_token": "selected-token"}),
            mark_text_used=mock.Mock(),
        )
        sys.modules["services.account_service"] = _account_service_module(account_service)
        spec = resolve_model("grok-imagine-image-edit")

        with mock.patch.object(grok, "GrokAppChatClient") as client_class:
            client = client_class.return_value.__enter__.return_value
            client.upload_image_edit_reference.return_value = grok.GrokImageEditReference("file-1", "https://assets.grok.com/input.png")
            client.create_image_edit_parent_post.return_value = ("post-1", "edit")
            client.stream_image_edit_events.return_value = iter([{
                "result": {
                    "response": {
                        "streamingImageGenerationResponse": {
                            "progress": 100,
                            "moderated": True,
                            "message": "Image was moderated",
                        }
                    }
                }
            }])
            with self.assertRaises(ImageGenerationError) as context:
                list(grok.app_chat_image_edit_outputs(
                    {"prompt": "edit"},
                    spec,
                    "edit",
                    [(b"png", "input.png", "image/png")],
                    1,
                    "1024x1024",
                ))

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.code, "content_policy_violation")
        self.assertIn("moderated", str(context.exception).lower())
        account_service.mark_text_used.assert_not_called()

    def test_grok_image_edit_validates_limits_as_openai_request_errors(self) -> None:
        spec = resolve_model("grok-imagine-image-edit")
        cases = [
            ([(b"png", "input.png", "image/png")], 1, "1792x1024", "size"),
            ([(b"png", "input.png", "image/png")], 3, "1024x1024", "n"),
            ([(b"png", f"input-{i}.png", "image/png") for i in range(8)], 1, "1024x1024", "image"),
        ]
        for images, n, size, param in cases:
            with self.subTest(param=param):
                with self.assertRaises(ImageGenerationError) as context:
                    list(grok.app_chat_image_edit_outputs({"prompt": "edit"}, spec, "edit", images, n, size))
                self.assertEqual(context.exception.status_code, 400)
                self.assertEqual(context.exception.error_type, "invalid_request_error")
                self.assertEqual(context.exception.param, param)

    def test_app_chat_error_classification_is_specific(self) -> None:
        cases = {
            401: (401, "upstream error"),
            402: (429, "rate limited"),
            403: (403, "forbidden"),
            429: (429, "rate limited"),
        }
        for upstream_status, (openai_status, message) in cases.items():
            with self.subTest(upstream_status=upstream_status):
                error = grok.classify_app_chat_upstream_error(upstream_status)

                self.assertEqual(error.status_code, openai_status)
                self.assertEqual(error.upstream_status, upstream_status)
                self.assertIn(message, str(error))

    def test_app_chat_error_classification_distinguishes_auth_challenge_limit_and_transient(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())

        class FakeResponse:
            def __init__(self, status_code: int, text: str = "", payload: dict[str, Any] | None = None) -> None:
                self.status_code = status_code
                self.text = text
                self._payload = payload

            def json(self) -> dict[str, Any]:
                if self._payload is None:
                    raise ValueError("not json")
                return self._payload

        with mock.patch.dict(sys.modules, {"services.account_service": types.SimpleNamespace(account_service=account_service)}):
            auth = grok.classify_app_chat_upstream_error(401, "sso=secret-token", FakeResponse(401, payload={"error": {"code": "unauthenticated"}}))
            challenge = grok.classify_app_chat_upstream_error(401, "sso=secret-token", FakeResponse(401, "<html>Just a moment... Cloudflare sso=leaked</html>"))
            limited = grok.classify_app_chat_upstream_error(400, "sso=secret-token", FakeResponse(400, payload={"error": {"code": "rate_limit_exceeded"}}))
            transient = grok.classify_app_chat_upstream_error(500, "sso=secret-token", FakeResponse(500, "HTTP2 stream reset"))

        self.assertEqual(auth.status_code, 401)
        self.assertEqual(auth.code, "authentication_failed")
        self.assertEqual(challenge.status_code, 502)
        self.assertEqual(challenge.code, "cloudflare_challenge")
        self.assertNotIn("secret-token", str(challenge))
        self.assertEqual(limited.status_code, 429)
        self.assertEqual(limited.code, "rate_limit_exceeded")
        self.assertEqual(transient.status_code, 502)
        self.assertEqual(transient.code, "upstream_transient")
        self.assertEqual(account_service.update_account.mock_calls, [
            mock.call("sso=secret-token", {"status": "异常"}),
            mock.call("sso=secret-token", {"status": "限流"}),
        ])

    def test_app_chat_403_classification_does_not_say_unsupported_model(self) -> None:
        error = grok.classify_app_chat_upstream_error(403)

        self.assertNotIn("unsupported", str(error).lower())
        self.assertNotIn("secret", str(error).lower())
        self.assertNotIn("browser", str(error).lower())
        self.assertNotIn("cf_clearance", str(error).lower())

    def test_app_chat_completion_uses_model_aware_account_selection(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(return_value="selected-token"),
            get_account=mock.Mock(return_value={"access_token": "selected-token", "cf_cookies": "cf_bm=account-bm"}),
            mark_text_used=mock.Mock(),
        )
        sys.modules["services.account_service"] = _account_service_module(account_service)
        spec = resolve_model("grok-4.20-heavy")

        with mock.patch.object(grok, "GrokAppChatClient") as client_class:
            client = client_class.return_value.__enter__.return_value
            client.stream_events.return_value = iter([{"result": {"response": {"token": "Hi", "messageTag": "final"}}}])
            events = list(grok.app_chat_completion_events({}, spec, [{"role": "user", "content": "Hello"}]))

        account_service.get_grok_app_chat_access_token.assert_called_once_with(spec, excluded_tokens=set())
        account_service.get_account.assert_called_once_with("selected-token")
        client_class.assert_called_once_with("selected-token", {"access_token": "selected-token", "cf_cookies": "cf_bm=account-bm"})
        self.assertEqual(events[0]["result"]["response"]["token"], "Hi")
        account_service.mark_text_used.assert_called_once_with("selected-token")

    def test_app_chat_completion_wraps_structured_grok_error_detail(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_app_chat_access_token=mock.Mock(return_value="selected-token"),
            get_account=mock.Mock(return_value={"access_token": "selected-token"}),
            mark_text_used=mock.Mock(),
        )
        sys.modules["services.account_service"] = _account_service_module(account_service)
        spec = resolve_model("grok-4.20-heavy")
        error = grok.GrokConsoleError(
            "Grok app-chat navigation timed out via Browser Bridge",
            504,
            504,
            code="grok_navigation_timeout",
            extra_detail={"bridge_code": "navigation_timeout"},
        )

        with mock.patch.object(grok, "GrokAppChatClient") as client_class:
            client = client_class.return_value.__enter__.return_value
            client.stream_events.side_effect = error
            with self.assertRaises(HTTPException) as ctx:
                list(grok.app_chat_completion_events({}, spec, [{"role": "user", "content": "Hello"}]))

        self.assertEqual(ctx.exception.status_code, 504)
        self.assertEqual(ctx.exception.detail["code"], "grok_navigation_timeout")
        self.assertEqual(ctx.exception.detail["bridge_code"], "navigation_timeout")
        self.assertEqual(ctx.exception.detail["error"], "Grok app-chat navigation timed out via Browser Bridge")
        account_service.mark_text_used.assert_not_called()

    def test_app_chat_headers_use_account_metadata_over_global_profile(self) -> None:
        with mock.patch.object(grok, "_grok_app_chat_profile", return_value=types.SimpleNamespace(
            user_agent="Global UA",
            cf_clearance="global-clearance",
            cf_cookies="cf_bm=global-bm",
            sec_ch_ua="global sec ua",
            sec_ch_ua_mobile="?0",
            sec_ch_ua_platform='"Windows"',
            statsig_id="global-statsig",
        )):
            headers = grok.app_chat_headers("selected-token", {
                "user_agent": "Account UA",
                "cf_cookies": "cf_bm=account-bm; cf_clearance=account-cookie-clearance",
                "cf_clearance": "account-clearance",
                "sec_ch_ua": "account sec ua",
                "sec_ch_ua_mobile": "?1",
                "sec_ch_ua_platform": '"Linux"',
            })

        self.assertEqual(headers["User-Agent"], "Account UA")
        self.assertEqual(headers["Sec-Ch-Ua"], "account sec ua")
        self.assertEqual(headers["Sec-Ch-Ua-Mobile"], "?1")
        self.assertEqual(headers["Sec-Ch-Ua-Platform"], '"Linux"')
        self.assertEqual(headers["x-statsig-id"], "global-statsig")
        self.assertEqual(headers["Cookie"], "sso=selected-token; sso-rw=selected-token; cf_bm=account-bm; cf_clearance=account-clearance")

    def test_grok_app_chat_validate_rate_limits_marks_selected_account_limited(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())

        class FakeResponse:
            status_code = 402

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with (
            mock.patch.dict(sys.modules, {"services.account_service": types.SimpleNamespace(account_service=account_service)}),
            mock.patch.object(grok.config, "data", {}),
            mock.patch("curl_cffi.requests.Session", FakeSession),
        ):
            client = grok.GrokAppChatClient("secret-token")
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                client.validate_rate_limits()

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.upstream_status, 402)
        account_service.update_account.assert_called_once_with("secret-token", {"status": "限流"})

    def test_grok_app_chat_validate_rate_limits_sanitizes_request_errors(self) -> None:
        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> object:
                raise grok.requests.exceptions.RequestException("failed with Cookie: sso=secret-token")

            def close(self) -> None:
                pass

        with mock.patch.object(grok.config, "data", {}), mock.patch("curl_cffi.requests.Session", FakeSession):
            client = grok.GrokAppChatClient("secret-token")
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                client.validate_rate_limits()

        self.assertEqual(str(ctx.exception), "Grok app-chat rate-limit check unavailable")
        self.assertEqual(ctx.exception.code, "rate_limit_network_error")
        self.assertTrue(ctx.exception.is_check_unavailable)
        self.assertNotIn("secret-token", str(ctx.exception))

    def test_grok_app_chat_validate_rate_limits_sanitizes_invalid_json(self) -> None:
        class FakeResponse:
            status_code = 200

            def json(self) -> object:
                raise ValueError("invalid json near sso=secret-token")

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with mock.patch.object(grok.config, "data", {}), mock.patch("curl_cffi.requests.Session", FakeSession):
            client = grok.GrokAppChatClient("secret-token")
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                client.validate_rate_limits()

        self.assertEqual(str(ctx.exception), "Grok app-chat rate-limit validation returned an invalid response")
        self.assertEqual(ctx.exception.code, "invalid_rate_limit_response")
        self.assertTrue(ctx.exception.is_check_unavailable)
        self.assertNotIn("secret-token", str(ctx.exception))

    def test_grok_app_chat_client_uses_account_impersonate_without_leaking_to_console_headers(self) -> None:
        settings = {
            "network_profiles": {
                "grok_console": {"user-agent": "Console UA", "cf_clearance": "console-clearance"},
                "grok_app_chat": {"impersonate": "global-browser", "user-agent": "Global UA"},
            }
        }
        created: list[dict[str, object]] = []

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                created.append(kwargs)

            def close(self) -> None:
                pass

        with mock.patch.object(grok.config, "data", settings), mock.patch("curl_cffi.requests.Session", FakeSession):
            client = grok.GrokAppChatClient("selected-token", {
                "browser": "account-browser",
                "user_agent": "Account UA",
                "cf_cookies": "cf_bm=account-bm",
            })
            app_headers = grok.app_chat_headers("selected-token", client.account)
            console_headers = grok._headers("selected-token")

        self.assertEqual(created, [{"impersonate": "account-browser", "verify": True}])
        self.assertEqual(app_headers["User-Agent"], "Account UA")
        self.assertIn("cf_bm=account-bm", app_headers["Cookie"])
        self.assertEqual(console_headers["User-Agent"], "Console UA")
        self.assertEqual(console_headers["Cookie"], "sso=selected-token; cf_clearance=console-clearance")

    def test_grok_console_default_network_profile_matches_existing_behavior(self) -> None:
        with mock.patch.object(grok.config, "data", {}):
            headers = grok._headers("token-value")

        self.assertEqual(headers["User-Agent"], "Mozilla/5.0 (webchat2api grok console)")
        self.assertEqual(headers["Cookie"], "sso=token-value")
        self.assertEqual(headers["Authorization"], "Bearer token-value")

    def test_grok_console_default_session_uses_network_profile(self) -> None:
        created: list[dict[str, object]] = []

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                created.append(kwargs)

            def close(self) -> None:
                pass

        with mock.patch.object(grok.config, "data", {}), mock.patch("curl_cffi.requests.Session", FakeSession):
            client = grok.GrokConsoleClient("token-value")

        self.assertEqual(created, [{"impersonate": "edge101", "verify": True}])
        self.assertEqual(client.network_profile.timeout, 60)

    def test_grok_console_stream_response_parses_sse_lines(self) -> None:
        calls: list[dict[str, object]] = []
        closed: list[bool] = []

        class FakeResponse:
            status_code = 200

            def iter_lines(self):
                return iter([
                    b": keepalive",
                    b"event: response.output_text.delta",
                    b'data: {"type":"response.output_text.delta","delta":"Hi"}',
                    b"data: [DONE]",
                    b'data: {"type":"response.output_text.delta","delta":" ignored"}',
                ])

            def close(self) -> None:
                closed.append(True)

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                calls.append({"url": url, **kwargs})
                return FakeResponse()

            def close(self) -> None:
                pass

        with mock.patch.object(grok.config, "data", {}), mock.patch("curl_cffi.requests.Session", FakeSession):
            client = grok.GrokConsoleClient("token-value")
            events = list(client.stream_response({"model": "grok-4.3", "input": []}))

        self.assertEqual(events, [{"type": "response.output_text.delta", "delta": "Hi"}])
        self.assertEqual(calls[0]["url"], grok.CONSOLE_RESPONSES_URL)
        self.assertTrue(calls[0]["stream"])
        self.assertEqual(cast(dict[str, Any], calls[0]["json"])["stream"], True)
        self.assertEqual(closed, [True])

    def test_grok_console_stream_response_uses_sse_event_name_when_data_has_no_type(self) -> None:
        class FakeResponse:
            status_code = 200

            def iter_lines(self):
                return iter([
                    b"event: response.reasoning_summary_text.delta",
                    b'data: {"delta":"think"}',
                    b"event: response.output_text.delta",
                    b'data: {"delta":"Hi"}',
                    b"data: [DONE]",
                ])

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with mock.patch.object(grok.config, "data", {}), mock.patch("curl_cffi.requests.Session", FakeSession):
            client = grok.GrokConsoleClient("token-value")
            events = list(client.stream_response({"model": "grok-4.3", "input": []}))

        self.assertEqual(
            events,
            [
                {"type": "response.reasoning_summary_text.delta", "delta": "think"},
                {"type": "response.output_text.delta", "delta": "Hi"},
            ],
        )
        self.assertEqual(grok.extract_console_stream_delta(events[0]).reasoning_content, "think")
        self.assertEqual(grok.extract_console_stream_delta(events[1]).content, "Hi")

    def test_grok_console_stream_response_aggregates_multiline_sse_data(self) -> None:
        class FakeResponse:
            status_code = 200

            def iter_lines(self):
                return iter([
                    b"event: response.output_text.delta",
                    b'data: {"delta":',
                    b'data: "Hi"}',
                    b"",
                    b"data:",
                    b"",
                    b"data: [DONE]",
                ])

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with mock.patch.object(grok.config, "data", {}), mock.patch("curl_cffi.requests.Session", FakeSession):
            client = grok.GrokConsoleClient("token-value")
            events = list(client.stream_response({"model": "grok-4.3", "input": []}))

        self.assertEqual(events, [{"type": "response.output_text.delta", "delta": "Hi"}])
        self.assertEqual(grok.extract_console_stream_delta(events[0]).content, "Hi")

    def test_grok_console_stream_response_resets_sse_event_after_dispatch(self) -> None:
        class FakeResponse:
            status_code = 200

            def iter_lines(self):
                return iter([
                    b"event: response.reasoning_summary_text.delta",
                    b'data: {"delta":"think"}',
                    b"",
                    b'data: {"delta":"plain"}',
                    b"",
                    b"data: [DONE]",
                ])

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with mock.patch.object(grok.config, "data", {}), mock.patch("curl_cffi.requests.Session", FakeSession):
            client = grok.GrokConsoleClient("token-value")
            events = list(client.stream_response({"model": "grok-4.3", "input": []}))

        self.assertEqual(events[0], {"type": "response.reasoning_summary_text.delta", "delta": "think"})
        self.assertEqual(events[1], {"delta": "plain"})
        self.assertEqual(grok.extract_console_stream_delta(events[0]).reasoning_content, "think")
        self.assertEqual(grok.extract_console_stream_delta(events[1]).content, "plain")

    def test_grok_console_stream_does_not_mark_reserved_quota_when_generator_is_closed(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_console_access_token=mock.Mock(return_value="selected-token"),
            get_account=mock.Mock(return_value={}),
            mark_grok_console_used=mock.Mock(),
        )
        spec = resolve_model("grok-4.3")

        class FakeClient:
            def __init__(self, access_token: str, account=None) -> None:
                self.access_token = access_token

            def __enter__(self) -> "FakeClient":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                pass

            def stream_response(self, payload):
                yield {"type": "response.output_text.delta", "delta": "Hi"}
                yield {"type": "response.output_text.delta", "delta": " later"}

        with (
            mock.patch.dict(sys.modules, {"services.account_service": types.SimpleNamespace(account_service=account_service)}),
            mock.patch.object(grok, "GrokConsoleClient", FakeClient),
        ):
            events = grok.console_chat_completion_events(
                {"model": "grok-4.3"},
                spec,
                [{"role": "user", "content": "Hello"}],
            )
            self.assertEqual(next(events), {"type": "response.output_text.delta", "delta": "Hi"})
            cast(Any, events).close()

        account_service.mark_grok_console_used.assert_not_called()

    def test_grok_console_stream_does_not_mark_reserved_quota_when_stream_completes_without_events(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_console_access_token=mock.Mock(return_value="selected-token"),
            get_account=mock.Mock(return_value={}),
            mark_grok_console_used=mock.Mock(),
        )
        spec = resolve_model("grok-4.3")

        class FakeClient:
            def __init__(self, access_token: str, account=None) -> None:
                self.access_token = access_token

            def __enter__(self) -> "FakeClient":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                pass

            def stream_response(self, payload):
                return iter(())

        with (
            mock.patch.dict(sys.modules, {"services.account_service": types.SimpleNamespace(account_service=account_service)}),
            mock.patch.object(grok, "GrokConsoleClient", FakeClient),
        ):
            events = list(grok.console_chat_completion_events(
                {"model": "grok-4.3"},
                spec,
                [{"role": "user", "content": "Hello"}],
            ))

        self.assertEqual(events, [])
        account_service.mark_grok_console_used.assert_not_called()

    def test_grok_console_stream_marks_account_used_after_partial_stream_error(self) -> None:
        account_service = types.SimpleNamespace(
            get_grok_console_access_token=mock.Mock(return_value="selected-token"),
            get_account=mock.Mock(return_value={}),
            mark_grok_console_used=mock.Mock(),
        )
        spec = resolve_model("grok-4.3")

        class FakeClient:
            def __init__(self, access_token: str, account=None) -> None:
                self.access_token = access_token

            def __enter__(self) -> "FakeClient":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                pass

            def stream_response(self, payload):
                yield {"type": "response.output_text.delta", "delta": "Hi"}
                raise grok.GrokConsoleError("stream failed", 502)

        with (
            mock.patch.dict(sys.modules, {"services.account_service": types.SimpleNamespace(account_service=account_service)}),
            mock.patch.object(grok, "GrokConsoleClient", FakeClient),
        ):
            events = grok.console_chat_completion_events(
                {"model": "grok-4.3"},
                spec,
                [{"role": "user", "content": "Hello"}],
            )
            self.assertEqual(next(events), {"type": "response.output_text.delta", "delta": "Hi"})
            with self.assertRaises(grok.HTTPException):
                next(events)

        account_service.mark_grok_console_used.assert_called_once_with("selected-token", success=False)

    def test_grok_console_stream_response_raises_stream_errors(self) -> None:
        class FakeResponse:
            status_code = 200

            def iter_lines(self):
                return iter([
                    b'data: {"type":"response.failed","error":{"message":"upstream failed"}}',
                ])

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with mock.patch.object(grok.config, "data", {}), mock.patch("curl_cffi.requests.Session", FakeSession):
            client = grok.GrokConsoleClient("token-value")
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                list(client.stream_response({"model": "grok-4.3", "input": []}))

        self.assertIn("upstream failed", str(ctx.exception))

    def test_grok_console_stream_response_includes_upstream_error_detail(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())

        class FakeResponse:
            status_code = 402

            def json(self):
                return {"error": {"message": "quota exhausted"}}

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                pass

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                return FakeResponse()

            def close(self) -> None:
                pass

        with (
            mock.patch.dict(sys.modules, {"services.account_service": types.SimpleNamespace(account_service=account_service)}),
            mock.patch.object(grok.config, "data", {}),
            mock.patch("curl_cffi.requests.Session", FakeSession),
        ):
            client = grok.GrokConsoleClient("token-value")
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                list(client.stream_response({"model": "grok-4.3", "input": []}))

        self.assertIn("quota exhausted", str(ctx.exception))
        account_service.update_account.assert_called_once_with("token-value", {"status": "限流"})

    def test_grok_console_uses_configured_network_profile(self) -> None:
        settings = {
            "network_profiles": {
                "grok_console": {
                    "impersonate": "chrome136",
                    "user-agent": "Configured Grok UA",
                    "verify": False,
                    "timeout": 12.5,
                }
            }
        }
        created: list[dict[str, object]] = []

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                created.append(kwargs)

            def close(self) -> None:
                pass

        with mock.patch.object(grok.config, "data", settings), mock.patch("curl_cffi.requests.Session", FakeSession):
            client = grok.GrokConsoleClient("sso=configured-cookie")
            headers = grok._headers("sso=configured-cookie")

        self.assertEqual(created, [{"impersonate": "chrome136", "verify": False}])
        self.assertEqual(client.network_profile.timeout, 12.5)
        self.assertEqual(headers["User-Agent"], "Configured Grok UA")
        self.assertEqual(headers["Cookie"], "sso=configured-cookie")

    def test_grok_app_chat_uses_dedicated_network_profile(self) -> None:
        settings = {
            "network_profiles": {
                "grok_console": {
                    "impersonate": "console-browser",
                    "user-agent": "Console UA",
                    "timeout": 11,
                },
                "grok_app_chat": {
                    "impersonate": "app-browser",
                    "user-agent": "App UA",
                    "verify": False,
                    "timeout": 7,
                    "cf_clearance": "app-clearance",
                },
            },
        }
        created: list[dict[str, object]] = []

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                created.append(kwargs)

            def close(self) -> None:
                pass

        with mock.patch.object(grok.config, "data", settings), mock.patch("curl_cffi.requests.Session", FakeSession):
            client = grok.GrokAppChatClient("token-value")
            headers = grok.app_chat_headers("token-value")

        self.assertEqual(created, [{"impersonate": "app-browser", "verify": False}])
        self.assertEqual(client.network_profile.timeout, 7)
        self.assertEqual(headers["User-Agent"], "App UA")
        self.assertEqual(headers["Cookie"], "sso=token-value; sso-rw=token-value; cf_clearance=app-clearance")

    def test_grok_console_session_preserves_proxy_kwargs(self) -> None:
        created: list[dict[str, object]] = []

        class FakeSession:
            headers: dict[str, str] = {}

            def __init__(self, **kwargs: object) -> None:
                created.append(kwargs)

            def close(self) -> None:
                pass

        with (
            mock.patch.object(grok.config, "data", {}),
            mock.patch.object(grok.config, "get_proxy_settings", return_value="http://proxy.local:8080"),
            mock.patch("curl_cffi.requests.Session", FakeSession),
        ):
            grok.GrokConsoleClient("token-value")

        self.assertEqual(created, [{"impersonate": "edge101", "verify": True, "proxy": "http://proxy.local:8080"}])

    def test_app_chat_status_feedback_updates_account(self) -> None:
        account_service = types.SimpleNamespace(update_account=mock.Mock())
        with mock.patch.dict(sys.modules, {"services.account_service": types.SimpleNamespace(account_service=account_service)}):
            for upstream_status in (401, 402, 403, 429):
                with self.subTest(upstream_status=upstream_status):
                    error = grok.classify_app_chat_upstream_error(upstream_status, "token-value")
                    expected_status = 429 if upstream_status == 402 else upstream_status
                    self.assertEqual(error.status_code, expected_status)

        self.assertEqual(account_service.update_account.mock_calls, [
            mock.call("token-value", {"status": "限流"}),
            mock.call("token-value", {"status": "限流"}),
        ])

    def test_app_chat_request_exception_messages_redact_credentials(self) -> None:
        message = "failed with Cookie: sso=secret-token; cf_clearance=secret-clearance; Authorization: Bearer secret-bearer"
        error = grok.GrokConsoleError(f"Grok app-chat upstream request failed: {grok._safe_exception_message(RuntimeError(message))}", 502)

        detail = error.to_http_detail()
        self.assertIn("sso=[redacted]", detail["error"])
        self.assertIn("cf_clearance=[redacted]", detail["error"])
        self.assertIn("Authorization: Bearer [redacted]", detail["error"])
        self.assertNotIn("secret-token", detail["error"])
        self.assertNotIn("secret-clearance", detail["error"])
        self.assertNotIn("secret-bearer", detail["error"])

    def test_grok_app_chat_uses_flaresolverr_on_403_then_retries(self) -> None:
        class FakeResponse:
            def __init__(self, status_code: int, lines: list[bytes] | None = None) -> None:
                self.status_code = status_code
                self._lines = lines or []

            def iter_lines(self):
                return iter(self._lines)

        class FakeSession:
            def __init__(self, **kwargs: object) -> None:
                self.headers: dict[str, str] = {}
                self.calls: list[dict[str, object]] = []
                self.responses = [
                    FakeResponse(403),
                    FakeResponse(200, [b'data: {"result":{"response":{"token":"ok","messageTag":"final"}}}']),
                ]

            def post(self, url: str, **kwargs: object) -> FakeResponse:
                self.calls.append({"url": url, **kwargs})
                return self.responses.pop(0)

            def close(self) -> None:
                pass

        settings = {
            "flaresolverr_url": "http://solver.local",
            "network_profiles": {"grok_app_chat": {"user-agent": "Old UA", "cf_clearance": "old-clearance"}},
        }
        clearance = types.SimpleNamespace(
            user_agent="Solved UA",
            cf_clearance="solved-clearance",
            cf_cookies="cf_clearance=solved-clearance; __cf_bm=solved-bm; x-challenge=solved-challenge",
        )
        updates: list[dict[str, object]] = []

        def fake_update(data: dict[str, object]) -> dict[str, object]:
            updates.append(data)
            settings.update(data)
            return settings

        with (
            mock.patch.object(grok.config, "data", settings),
            mock.patch.object(grok.config, "update", side_effect=fake_update) as update,
            mock.patch.object(grok.FlareSolverrClearanceProvider, "solve", return_value=clearance) as solve,
            mock.patch.object(grok, "create_session", return_value=FakeSession()),
        ):
            client = grok.GrokAppChatClient("token-value")
            events = list(client.stream_events({"message": "hi"}))

        solve.assert_called_once_with()
        update.assert_called_once()
        self.assertEqual(len(client.session.calls), 2)
        retry_headers = client.session.calls[1]["headers"]
        self.assertEqual(retry_headers["User-Agent"], "Solved UA")
        self.assertIn("cf_clearance=solved-clearance", retry_headers["Cookie"])
        self.assertEqual(client.session.headers["User-Agent"], "Solved UA")
        self.assertEqual(events[0]["result"]["response"]["token"], "ok")
        saved_profiles = cast(dict[str, Any], updates[0]["network_profiles"])
        saved_profile = cast(dict[str, str], saved_profiles["grok_app_chat"])
        self.assertEqual(saved_profile["user-agent"], "Solved UA")
        self.assertEqual(saved_profile["cf_clearance"], "solved-clearance")
        self.assertEqual(saved_profile["cf_cookies"], "cf_clearance=solved-clearance; __cf_bm=solved-bm; x-challenge=solved-challenge")

    def test_grok_app_chat_refresh_derives_coherent_browser_and_headers(self) -> None:
        class FakeSession:
            headers: dict[str, str] = {}

            def close(self) -> None:
                pass

        settings = {
            "flaresolverr_url": "http://solver.local",
            "network_profiles": {"grok_app_chat": {"user-agent": "Old UA", "browser": "chrome136"}},
        }
        clearance = types.SimpleNamespace(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
            cf_clearance="solved-clearance",
            cf_cookies="cf_clearance=solved-clearance; __cf_bm=solved-bm",
        )
        updates: list[dict[str, object]] = []

        def fake_update(data: dict[str, object]) -> dict[str, object]:
            updates.append(data)
            settings.update(data)
            return settings

        with (
            mock.patch.object(grok.config, "data", settings),
            mock.patch.object(grok.config, "update", side_effect=fake_update),
            mock.patch.object(grok.FlareSolverrClearanceProvider, "solve", return_value=clearance),
            mock.patch.object(grok, "create_session", return_value=FakeSession()),
        ):
            client = grok.GrokAppChatClient("token-value")
            self.assertTrue(client._refresh_clearance())
            headers = grok.app_chat_headers("token-value")

        saved_profiles = cast(dict[str, Any], updates[0]["network_profiles"])
        saved_profile = cast(dict[str, str], saved_profiles["grok_app_chat"])
        self.assertEqual(saved_profile["browser"], "chrome141")
        self.assertEqual(saved_profile["impersonate"], "chrome141")
        self.assertEqual(headers["User-Agent"], clearance.user_agent)
        self.assertEqual(headers["Sec-Ch-Ua"], '"Chromium";v="141", "Google Chrome";v="141", "Not.A/Brand";v="99"')
        self.assertEqual(headers["Sec-Ch-Ua-Mobile"], "?0")
        self.assertEqual(headers["Sec-Ch-Ua-Platform"], '"Windows"')

    def test_flaresolverr_provider_posts_solution_request_with_proxy(self) -> None:
        class FakeResponse:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, object]:
                return {
                    "solution": {
                        "userAgent": "Solved UA",
                        "cookies": [
                            {"name": "cf_clearance", "value": "clearance-value"},
                            {"name": "__cf_bm", "value": "bm-value"},
                            {"name": "session", "value": "kept"},
                            {"name": "x-challenge", "value": "challenge-value"},
                            {"name": "x-signature", "value": "signature-value"},
                            {"name": "empty", "value": ""},
                            {"name": "bad;name", "value": "bad-value"},
                            {"name": "bad-value", "value": "line\nbreak"},
                        ],
                    }
                }

        with (
            mock.patch.object(flaresolverr.config, "data", {"flaresolverr_url": "http://solver.local", "flaresolverr_timeout_sec": 12}),
            mock.patch.object(flaresolverr.config, "get_proxy_settings", return_value="http://proxy.local:8080"),
            mock.patch.object(flaresolverr.requests, "post", return_value=FakeResponse(), create=True) as post,
        ):
            clearance = flaresolverr.FlareSolverrClearanceProvider().solve()

        post.assert_called_once_with(
            "http://solver.local/v1",
            json={
                "cmd": "request.get",
                "url": "https://grok.com",
                "maxTimeout": 12000,
                "proxy": {"url": "http://proxy.local:8080"},
            },
            timeout=17,
        )
        self.assertIsNotNone(clearance)
        assert clearance is not None
        self.assertEqual(clearance.user_agent, "Solved UA")
        self.assertEqual(clearance.cf_clearance, "clearance-value")
        self.assertEqual(
            clearance.cf_cookies,
            "cf_clearance=clearance-value; __cf_bm=bm-value; session=kept; x-challenge=challenge-value; x-signature=signature-value",
        )


class TestBrowserBridge(unittest.TestCase):
    def test_extract_raw_sso_plain_token(self):
        sso = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test"
        self.assertEqual(grok._extract_raw_sso(sso), sso)

    def test_extract_raw_sso_with_prefix(self):
        self.assertEqual(grok._extract_raw_sso("sso=abc123"), "abc123")

    def test_extract_raw_sso_from_cookie_header(self):
        self.assertEqual(grok._extract_raw_sso("sso=abc123; cf_clearance=xyz; other=val"), "abc123")

    def test_extract_raw_sso_empty(self):
        self.assertEqual(grok._extract_raw_sso(""), "")
        self.assertEqual(grok._extract_raw_sso(cast(str, None)), "")

    @mock.patch("services.providers.grok._detect_bridge_url", return_value="")
    def test_try_browser_bridge_returns_none_when_no_bridge(self, _):
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client.access_token = "test_sso"
        self.assertIsNone(client._try_browser_bridge({"message": "hi"}))

    @mock.patch("services.providers.grok._detect_bridge_url", return_value="http://127.0.0.1:3080")
    def test_try_browser_bridge_calls_bridge(self, _):
        resp = mock.MagicMock()
        resp.status = 200
        resp.read.return_value = b'{"result":{"response":{"token":"hi"}}}\n'
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: None
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client.access_token = "test_sso_token"
        with mock.patch("urllib.request.urlopen", return_value=resp):
            result = client._try_browser_bridge({"message": "test"})
            self.assertIsNotNone(result)
            assert result is not None
            self.assertGreater(len(result), 0)

    @mock.patch("services.providers.grok.config")
    def test_stream_events_does_not_auto_fallback_to_bridge_for_direct_403(self, mock_config):
        mock_config.browser_bridge_url = ""
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client._stream_direct_events = mock.Mock(side_effect=grok.GrokConsoleError("Grok app-chat forbidden (HTTP 403)", 403, 403))
        client._try_browser_bridge = mock.Mock(return_value=['{"result":{"response":{"token":"bridge"}}}'])

        with self.assertRaises(grok.GrokConsoleError) as ctx:
            list(client.stream_events({"message": "test"}))

        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(ctx.exception.upstream_status, 403)
        client._try_browser_bridge.assert_not_called()

    @mock.patch("services.providers.grok.config")
    def test_stream_events_auto_fallback_to_bridge_for_transient_direct_status(self, mock_config):
        mock_config.browser_bridge_url = ""
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client._stream_direct_events = mock.Mock(side_effect=grok.GrokConsoleError("Grok app-chat upstream timeout (HTTP 408)", 502, 408))
        client._try_browser_bridge = mock.Mock(return_value=['{"result":{"response":{"token":"bridge"}}}'])

        events = list(client.stream_events({"message": "test"}))

        self.assertEqual(events, [{"result": {"response": {"token": "bridge"}}}])
        client._try_browser_bridge.assert_called_once_with({"message": "test"})

    @mock.patch("services.providers.grok.config")
    def test_stream_events_configured_bridge_still_uses_bridge_first(self, mock_config):
        mock_config.browser_bridge_url = "http://bridge:3080"
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client._stream_direct_events = mock.Mock(return_value=iter(()))
        client._try_browser_bridge = mock.Mock(return_value=['{"result":{"response":{"token":"bridge"}}}'])

        events = list(client.stream_events({"message": "test"}))

        self.assertEqual(events, [{"result": {"response": {"token": "bridge"}}}])
        client._try_browser_bridge.assert_called_once_with({"message": "test"})
        client._stream_direct_events.assert_not_called()

    @mock.patch("services.providers.grok.config")
    def test_detect_bridge_url_uses_loopback_config_first(self, mock_config):
        mock_config.browser_bridge_url = "http://127.0.0.1:9999"
        grok._bridge_probed = False
        self.assertEqual(grok._detect_bridge_url(), "http://127.0.0.1:9999")

    @mock.patch("services.providers.grok.config")
    def test_detect_bridge_url_rejects_non_loopback_config(self, mock_config):
        mock_config.browser_bridge_url = "http://custom:9999"
        grok._bridge_probed = False
        with self.assertRaises(grok.GrokConsoleError) as ctx:
            grok._detect_bridge_url()
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.code, "grok_browser_bridge_url_not_loopback")

    @mock.patch("services.providers.grok._detect_bridge_url", return_value="http://127.0.0.1:3080")
    def test_try_browser_bridge_403_reports_tier_hint(self, _):
        import urllib.error
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client.access_token = "test_sso_token"
        err = urllib.error.HTTPError(
            url="http://127.0.0.1:3080/api/chat",
            code=403,
            msg="Forbidden",
            hdrs=cast(Any, None),
            fp=None,
        )
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                client._try_browser_bridge({"message": "test"})
        self.assertIn("account may lack required tier", str(ctx.exception))

    @mock.patch("services.providers.grok.config")
    def test_explicit_bridge_health_unavailable_fast_fails_with_structured_code(self, mock_config):
        import urllib.error
        mock_config.browser_bridge_url = "http://127.0.0.1:3080"
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client.access_token = "test_sso_token"

        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")) as urlopen:
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                client._try_browser_bridge({"message": "test"})

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.code, "grok_browser_bridge_unavailable")
        self.assertEqual(ctx.exception.to_http_detail()["bridge_code"], "browser_bridge_unavailable")
        self.assertIn("Browser Bridge unavailable", str(ctx.exception))
        self.assertEqual(urlopen.call_args.kwargs["timeout"], grok._BRIDGE_HEALTH_TIMEOUT)

    @mock.patch("services.providers.grok.config")
    def test_explicit_bridge_degraded_health_fast_fails_with_structured_code(self, mock_config):
        mock_config.browser_bridge_url = "http://127.0.0.1:3080"
        health = mock.MagicMock()
        health.read.return_value = json.dumps({
            "status": "degraded",
            "last_error_code": "bridge_unavailable",
            "last_error": "Chromium launch failed",
        }).encode()
        health.__enter__ = lambda s: s
        health.__exit__ = lambda s, *a: None
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client.access_token = "test_sso_token"

        with mock.patch("urllib.request.urlopen", return_value=health):
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                client._try_browser_bridge({"message": "test"})

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.code, "grok_browser_bridge_unavailable")
        self.assertEqual(ctx.exception.to_http_detail()["bridge_code"], "bridge_unavailable")
        self.assertIn("Browser Bridge unavailable", str(ctx.exception))

    @mock.patch("services.providers.grok.config")
    def test_explicit_bridge_ok_health_ignores_historical_last_error(self, mock_config):
        mock_config.browser_bridge_url = "http://127.0.0.1:3080"
        health = mock.MagicMock()
        health.read.return_value = json.dumps({
            "status": "ok",
            "last_error_code": "bridge_unavailable",
            "last_error": "Historical launch failure",
        }).encode()
        health.__enter__ = lambda s: s
        health.__exit__ = lambda s, *a: None
        chat = mock.MagicMock()
        chat.status = 200
        chat.read.return_value = b'{"result":{"response":{"token":"hi","messageTag":"final"}}}\n'
        chat.__enter__ = lambda s: s
        chat.__exit__ = lambda s, *a: None
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client.access_token = "test_sso_token"

        with mock.patch("urllib.request.urlopen", side_effect=[health, chat]) as urlopen:
            result = client._try_browser_bridge({"message": "test"})

        self.assertEqual(result, ['{"result":{"response":{"token":"hi","messageTag":"final"}}}'])
        self.assertEqual(len(urlopen.call_args_list), 2)

    @mock.patch("services.providers.grok.config")
    def test_explicit_bridge_navigation_timeout_maps_to_http_detail(self, mock_config):
        import urllib.error
        mock_config.browser_bridge_url = "http://127.0.0.1:3080"
        health = mock.MagicMock()
        health.read.return_value = b'{"status":"ok"}'
        health.__enter__ = lambda s: s
        health.__exit__ = lambda s, *a: None
        error_body = cast(Any, types.SimpleNamespace(read=lambda: b'{"error":"Navigation to grok.com timed out","code":"navigation_timeout"}'))
        http_error = urllib.error.HTTPError(
            url="http://127.0.0.1:3080/api/chat",
            code=504,
            msg="Gateway Timeout",
            hdrs=cast(Any, None),
            fp=error_body,
        )
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client.access_token = "test_sso_token"

        with mock.patch("urllib.request.urlopen", side_effect=[health, http_error]):
            with self.assertRaises(grok.HTTPException) as ctx:
                try:
                    client._try_browser_bridge({"message": "test"})
                except grok.GrokConsoleError as exc:
                    raise grok.HTTPException(status_code=exc.status_code, detail=exc.to_http_detail())

        self.assertEqual(ctx.exception.status_code, 504)
        self.assertEqual(ctx.exception.detail["code"], "grok_navigation_timeout")
        self.assertEqual(ctx.exception.detail["bridge_code"], "navigation_timeout")
        self.assertEqual(ctx.exception.detail["error"], "Grok app-chat navigation timed out via Browser Bridge")

    @mock.patch("services.providers.grok.config")
    def test_explicit_bridge_healthy_response_yields_app_chat_events(self, mock_config):
        mock_config.browser_bridge_url = "http://127.0.0.1:3080"
        health = mock.MagicMock()
        health.read.return_value = b'{"status":"ok","pages":0}'
        health.__enter__ = lambda s: s
        health.__exit__ = lambda s, *a: None
        chat = mock.MagicMock()
        chat.status = 200
        chat.read.return_value = b'{"result":{"response":{"token":"hi","messageTag":"final"}}}\n'
        chat.__enter__ = lambda s: s
        chat.__exit__ = lambda s, *a: None
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client.access_token = "test_sso_token"

        with mock.patch("urllib.request.urlopen", side_effect=[health, chat]) as urlopen:
            events = list(client.stream_events({"message": "test"}))

        self.assertEqual(events, [{"result": {"response": {"token": "hi", "messageTag": "final"}}}])
        self.assertEqual(urlopen.call_args_list[0].kwargs["timeout"], grok._BRIDGE_HEALTH_TIMEOUT)
        self.assertEqual(urlopen.call_args_list[1].kwargs["timeout"], grok._BRIDGE_EXPLICIT_CHAT_TIMEOUT)

    @mock.patch("services.providers.grok.config")
    def test_auto_detected_bridge_unavailable_still_returns_none(self, mock_config):
        import urllib.error
        mock_config.browser_bridge_url = ""
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        client.access_token = "test_sso_token"

        with (
            mock.patch("services.providers.grok._detect_bridge_url", return_value="http://127.0.0.1:3080"),
            mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")),
        ):
            self.assertIsNone(client._try_browser_bridge({"message": "test"}))

    @mock.patch("services.providers.grok.config")
    def test_app_chat_prefers_direct_when_bridge_is_auto_detected(self, mock_config):
        mock_config.browser_bridge_url = ""
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        direct_event = {"result": {"response": {"token": "hi"}}}
        with (
            mock.patch.object(client, "_stream_direct_events", return_value=iter([direct_event])) as direct,
            mock.patch.object(client, "_try_browser_bridge") as bridge,
        ):
            self.assertEqual(list(client.stream_events({"message": "test"})), [direct_event])
        direct.assert_called_once()
        bridge.assert_not_called()

    @mock.patch("services.providers.grok.config")
    def test_app_chat_uses_explicit_bridge_before_direct(self, mock_config):
        mock_config.browser_bridge_url = "http://127.0.0.1:3080"
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        with (
            mock.patch.object(client, "_try_browser_bridge", return_value=['{"result":{"response":{"token":"hi"}}}']) as bridge,
            mock.patch.object(client, "_stream_direct_events") as direct,
        ):
            events = list(client.stream_events({"message": "test"}))
        self.assertEqual(events, [{"result": {"response": {"token": "hi"}}}])
        bridge.assert_called_once()
        direct.assert_not_called()

    @mock.patch("services.providers.grok.config")
    def test_app_chat_does_not_fall_back_to_bridge_after_direct_403(self, mock_config):
        mock_config.browser_bridge_url = ""
        client = grok.GrokAppChatClient.__new__(grok.GrokAppChatClient)
        with (
            mock.patch.object(client, "_stream_direct_events", side_effect=grok.GrokConsoleError("forbidden", 403, 403)) as direct,
            mock.patch.object(client, "_try_browser_bridge", return_value=['{"result":{"response":{"token":"hi"}}}']) as bridge,
        ):
            with self.assertRaises(grok.GrokConsoleError) as ctx:
                list(client.stream_events({"message": "test"}))
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(ctx.exception.upstream_status, 403)
        direct.assert_called_once()
        bridge.assert_not_called()

    @mock.patch("services.providers.grok.config")
    def test_detect_bridge_url_auto_probes(self, mock_config):
        mock_config.browser_bridge_url = ""
        grok._bridge_probed = False
        grok._bridge_detected_url = None
        resp = mock.MagicMock()
        resp.status = 200
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: None
        with mock.patch("urllib.request.urlopen", return_value=resp):
            result = grok._detect_bridge_url()
            self.assertEqual(result, "http://127.0.0.1:3080")


if __name__ == "__main__":
    unittest.main()

