import asyncio
import io
import os
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

import ranker


class AsyncContextManager:
    def __init__(self, value, *, enter_error=None, exit_error=None):
        self.value = value
        self.enter_error = enter_error
        self.exit_error = exit_error
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        if self.enter_error is not None:
            raise self.enter_error
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        self.exited = True
        if self.exit_error is not None:
            raise self.exit_error
        return False


def make_panel_clients():
    return ranker.PanelClients(
        gpt=MagicMock(name="gpt_client"),
        gemini=MagicMock(name="gemini_client"),
        claude=MagicMock(name="claude_client"),
        grok=MagicMock(name="grok_client"),
    )


def make_panel_responses(label):
    return dict(
        gpt=f"GPT {label}",
        gemini=f"Gemini {label}",
        claude=f"Claude {label}",
        grok=f"Grok {label}",
    )


class ProviderInvocationTests(unittest.IsolatedAsyncioTestCase):
    async def test_gpt_returns_response_text(self):
        response = types.SimpleNamespace(
            error=None,
            status="completed",
            output_text="GPT result",
        )
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=response)

        result = await ranker.invoke_gpt(client, "gpt-test", "Question", ())

        self.assertEqual(result, "GPT result")

    async def test_gpt_rejects_non_completed_and_malformed_responses(self):
        responses = [
            (
                types.SimpleNamespace(
                    error=types.SimpleNamespace(
                        code="server_error", message="Model failure"
                    ),
                    status="failed",
                    output_text="",
                ),
                "GPT response status 'failed'",
            ),
            (
                types.SimpleNamespace(
                    error=None,
                    status="failed",
                    output_text="",
                ),
                "GPT response status 'failed'",
            ),
            (
                types.SimpleNamespace(error=None, output_text="GPT result"),
                "Error invoking OpenAI API",
            ),
        ]

        for response, expected_error in responses:
            with self.subTest(response=response):
                client = MagicMock()
                client.responses.create = AsyncMock(return_value=response)
                with self.assertRaisesRegex(
                    ranker.ModelInvocationError, expected_error
                ):
                    await ranker.invoke_gpt(client, "gpt-test", "Question", ())

    async def test_gpt_rejects_incomplete_and_blank_responses(self):
        responses = [
            (
                types.SimpleNamespace(
                    error=None,
                    status="incomplete",
                    output_text="Partial GPT result",
                ),
                "GPT response status 'incomplete'",
            ),
            (
                types.SimpleNamespace(
                    error=None,
                    status="completed",
                    incomplete_details=None,
                    output_text="  ",
                ),
                "GPT response contained no text",
            ),
        ]

        for response, expected_error in responses:
            with self.subTest(response=response):
                client = MagicMock()
                client.responses.create = AsyncMock(return_value=response)
                with self.assertRaisesRegex(
                    ranker.ModelInvocationError, expected_error
                ):
                    await ranker.invoke_gpt(client, "gpt-test", "Question", ())

    async def test_gpt_wraps_api_exceptions(self):
        client = MagicMock()
        client.responses.create = AsyncMock(
            side_effect=RuntimeError("connection failed")
        )

        with self.assertRaisesRegex(ranker.ModelInvocationError, "connection failed"):
            await ranker.invoke_gpt(client, "gpt-test", "Question", ())

    async def test_gemini_returns_response_text(self):
        response = types.SimpleNamespace(
            prompt_feedback=None,
            candidates=[
                types.SimpleNamespace(
                    finish_reason=ranker.types.FinishReason.STOP,
                    finish_message=None,
                )
            ],
            text="Gemini result",
        )
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(return_value=response)

        result = await ranker.invoke_gemini(client, "gemini-test", "Question", ())

        self.assertEqual(result, "Gemini result")

    async def test_grok_returns_stripped_response_text(self):
        response = types.SimpleNamespace(
            finish_reason="REASON_STOP",
            content="  Grok result  ",
        )
        chat = MagicMock()
        chat.sample = AsyncMock(return_value=response)
        client = MagicMock()
        client.chat.create.return_value = chat

        result = await ranker.invoke_grok(client, "grok-test", "Question", ())

        self.assertEqual(result, "Grok result")
        client.chat.create.assert_called_once_with(
            model="grok-test",
            tools=[],
            reasoning_effort="xhigh",
            service_tier="default",
        )

    async def test_grok_priority_requests_priority_service_tier(self):
        client = MagicMock()
        client.chat.create.return_value.sample = AsyncMock(
            return_value=types.SimpleNamespace(
                finish_reason="REASON_STOP", content="Grok result"
            )
        )

        result = await ranker.invoke_grok(
            client, "grok-test", "Question", (), priority=True
        )

        self.assertEqual(result, "Grok result")
        client.chat.create.assert_called_once_with(
            model="grok-test",
            tools=[],
            reasoning_effort="xhigh",
            service_tier="priority",
        )

    async def test_grok_rejects_unsuccessful_response_results(self):
        responses = [
            (
                types.SimpleNamespace(
                    finish_reason="REASON_MAX_LEN",
                    content="Partial Grok result",
                ),
                "stopped with 'REASON_MAX_LEN'",
            ),
            (
                types.SimpleNamespace(finish_reason="STOP", content="Grok result"),
                "stopped with 'STOP'",
            ),
            (
                types.SimpleNamespace(finish_reason="stop", content="Grok result"),
                "stopped with 'stop'",
            ),
            (
                types.SimpleNamespace(
                    finish_reason="REASON_STOP",
                    content="  ",
                ),
                "contained no text",
            ),
        ]
        chat = MagicMock()
        chat.sample = AsyncMock(side_effect=[response for response, _ in responses])
        client = MagicMock()
        client.chat.create.return_value = chat

        for _, expected_error in responses:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(
                    ranker.ModelInvocationError, expected_error
                ):
                    await ranker.invoke_grok(client, "grok-test", "Question", ())

    async def test_grok_wraps_malformed_responses(self):
        responses = [
            types.SimpleNamespace(content="Grok result"),
            types.SimpleNamespace(finish_reason="REASON_STOP"),
        ]

        for response in responses:
            with self.subTest(response=response):
                chat = MagicMock()
                chat.sample = AsyncMock(return_value=response)
                client = MagicMock()
                client.chat.create.return_value = chat

                with self.assertRaisesRegex(
                    ranker.ModelInvocationError, "Malformed Grok response"
                ) as raised:
                    await ranker.invoke_grok(client, "grok-test", "Question", ())

                self.assertIsInstance(raised.exception.__cause__, AttributeError)

    async def test_grok_wraps_setup_and_api_exceptions(self):
        client = MagicMock()
        client.chat.create.side_effect = RuntimeError("invalid model")
        with self.assertRaisesRegex(ranker.ModelInvocationError, "invalid model"):
            await ranker.invoke_grok(client, "grok-test", "Question", ())

        chat = MagicMock()
        chat.sample = AsyncMock(side_effect=RuntimeError("connection failed"))
        client = MagicMock()
        client.chat.create.return_value = chat
        with self.assertRaisesRegex(ranker.ModelInvocationError, "connection failed"):
            await ranker.invoke_grok(client, "grok-test", "Question", ())

    async def test_claude_returns_response_text(self):
        response = types.SimpleNamespace(
            stop_reason="end_turn",
            content=[types.SimpleNamespace(type="text", text="Claude result")],
        )
        stream = MagicMock()
        stream.get_final_message = AsyncMock(return_value=response)
        client = MagicMock()
        client.messages.stream.return_value = AsyncContextManager(stream)

        result = await ranker.invoke_claude(client, "claude-test", "Question", ())

        self.assertEqual(result, "Claude result")

    async def test_claude_builds_stream_request_with_history(self):
        response = types.SimpleNamespace(
            stop_reason="end_turn",
            content=[types.SimpleNamespace(type="text", text="Claude result")],
        )
        stream = MagicMock()
        stream.get_final_message = AsyncMock(return_value=response)
        client = MagicMock()
        client.messages.stream.return_value = AsyncContextManager(stream)
        history = (("Earlier question", "Earlier answer"),)

        await ranker.invoke_claude(client, "claude-test", "Question", history)

        client.messages.stream.assert_called_once_with(
            model="claude-test",
            max_tokens=32768,
            messages=[
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
                {"role": "user", "content": "Question"},
            ],
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        )

    async def test_claude_wraps_api_exceptions(self):
        client = MagicMock()
        error = RuntimeError("connection failed")
        client.messages.stream.side_effect = error

        with self.assertRaisesRegex(
            ranker.ModelInvocationError,
            "Error invoking Anthropic API: connection failed",
        ) as raised:
            await ranker.invoke_claude(client, "claude-test", "Question", ())

        self.assertIs(raised.exception.__cause__, error)

    async def test_claude_wraps_stream_lifecycle_exceptions(self):
        response = types.SimpleNamespace(
            stop_reason="end_turn",
            content=[types.SimpleNamespace(type="text", text="Claude result")],
        )
        cases = [
            (
                "final message",
                AsyncContextManager(MagicMock()),
                RuntimeError("read failed"),
            ),
            (
                "context entry",
                AsyncContextManager(
                    MagicMock(), enter_error=RuntimeError("enter failed")
                ),
                None,
            ),
            (
                "context exit",
                AsyncContextManager(
                    MagicMock(), exit_error=RuntimeError("exit failed")
                ),
                None,
            ),
        ]

        for name, manager, stream_error in cases:
            with self.subTest(name=name):
                stream = manager.value
                stream.get_final_message = AsyncMock(
                    side_effect=stream_error or response
                )
                client = MagicMock()
                client.messages.stream.return_value = manager

                with self.assertRaisesRegex(
                    ranker.ModelInvocationError,
                    "Error invoking Anthropic API",
                ) as raised:
                    await ranker.invoke_claude(client, "claude-test", "Question", ())

                self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    async def test_claude_propagates_cancellation(self):
        stream = MagicMock()
        stream.get_final_message = AsyncMock(side_effect=asyncio.CancelledError())
        client = MagicMock()
        client.messages.stream.return_value = AsyncContextManager(stream)

        with self.assertRaises(asyncio.CancelledError):
            await ranker.invoke_claude(client, "claude-test", "Question", ())

    async def test_claude_rejects_incomplete_or_textless_responses(self):
        responses = [
            (
                types.SimpleNamespace(
                    stop_reason="max_tokens",
                    content=[types.SimpleNamespace(type="text", text="Partial result")],
                ),
                "Claude response stopped with 'max_tokens'",
            ),
            (
                types.SimpleNamespace(
                    stop_reason="refusal",
                    content=[types.SimpleNamespace(type="text", text="Refusal")],
                ),
                "Claude response stopped with 'refusal'",
            ),
            (
                types.SimpleNamespace(
                    stop_reason="end_turn",
                    content=[types.SimpleNamespace(type="thinking")],
                ),
                "Claude response contained no text",
            ),
            (
                types.SimpleNamespace(
                    stop_reason="end_turn",
                    content=[types.SimpleNamespace(type="text", text="  ")],
                ),
                "Claude response contained no text",
            ),
        ]
        stream = MagicMock()
        stream.get_final_message = AsyncMock(
            side_effect=[response for response, _ in responses]
        )
        client = MagicMock()
        client.messages.stream.return_value = AsyncContextManager(stream)

        for _, expected_error in responses:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(
                    ranker.ModelInvocationError, expected_error
                ) as raised:
                    await ranker.invoke_claude(client, "claude-test", "Question", ())
                self.assertNotIn("Error invoking Anthropic API", str(raised.exception))

    async def test_claude_wraps_malformed_responses(self):
        responses = [
            types.SimpleNamespace(content=[]),
            types.SimpleNamespace(stop_reason="end_turn"),
            types.SimpleNamespace(stop_reason="end_turn", content=None),
            types.SimpleNamespace(
                stop_reason="end_turn",
                content=[types.SimpleNamespace(type="text")],
            ),
            types.SimpleNamespace(
                stop_reason="end_turn",
                content=[types.SimpleNamespace(type="text", text=123)],
            ),
        ]

        for response in responses:
            with self.subTest(response=response):
                stream = MagicMock()
                stream.get_final_message = AsyncMock(return_value=response)
                client = MagicMock()
                client.messages.stream.return_value = AsyncContextManager(stream)

                with self.assertRaisesRegex(
                    ranker.ModelInvocationError, "Malformed Claude response"
                ) as raised:
                    await ranker.invoke_claude(client, "claude-test", "Question", ())

                self.assertIsInstance(
                    raised.exception.__cause__, (AttributeError, TypeError)
                )

    async def test_gemini_rejects_api_and_response_failures(self):
        responses = [
            (
                types.SimpleNamespace(
                    prompt_feedback=types.SimpleNamespace(
                        block_reason=ranker.types.BlockedReason.SAFETY
                    ),
                    candidates=None,
                    text=None,
                ),
                "contained no candidates",
            ),
            (
                types.SimpleNamespace(
                    prompt_feedback=None,
                    candidates=None,
                    text=None,
                ),
                "contained no candidates",
            ),
            (
                types.SimpleNamespace(
                    prompt_feedback=None,
                    candidates=[
                        types.SimpleNamespace(
                            finish_reason=ranker.types.FinishReason.MAX_TOKENS,
                            finish_message="Output limit reached",
                        )
                    ],
                    text="Partial Gemini result",
                ),
                "stopped with 'MAX_TOKENS'",
            ),
            (
                types.SimpleNamespace(
                    prompt_feedback=None,
                    candidates=[
                        types.SimpleNamespace(
                            finish_reason=None,
                            finish_message=None,
                        )
                    ],
                    text="Partial Gemini result",
                ),
                "stopped with 'unknown reason'",
            ),
            (
                types.SimpleNamespace(
                    prompt_feedback=None,
                    candidates=[
                        types.SimpleNamespace(
                            finish_reason=ranker.types.FinishReason.STOP,
                            finish_message=None,
                        )
                    ],
                    text=None,
                ),
                "contained no text",
            ),
            (
                types.SimpleNamespace(
                    prompt_feedback=None,
                    candidates=[
                        types.SimpleNamespace(
                            finish_reason=ranker.types.FinishReason.STOP,
                            finish_message=None,
                        )
                    ],
                    text="  ",
                ),
                "contained no text",
            ),
        ]
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(
            side_effect=[response for response, _ in responses]
        )

        for _, expected_error in responses:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(
                    ranker.ModelInvocationError, expected_error
                ):
                    await ranker.invoke_gemini(client, "gemini-test", "Question", ())

    async def test_gemini_wraps_malformed_responses(self):
        responses = [
            types.SimpleNamespace(),
            types.SimpleNamespace(
                candidates=[
                    types.SimpleNamespace(
                        finish_reason=ranker.types.FinishReason.STOP,
                        finish_message=None,
                    )
                ]
            ),
            types.SimpleNamespace(
                candidates=[types.SimpleNamespace(finish_message=None)],
                text="Gemini result",
            ),
            types.SimpleNamespace(
                candidates=[
                    types.SimpleNamespace(
                        finish_reason=ranker.types.FinishReason.MAX_TOKENS
                    )
                ],
                text="Partial Gemini result",
            ),
        ]

        for response in responses:
            with self.subTest(response=response):
                client = MagicMock()
                client.aio.models.generate_content = AsyncMock(return_value=response)

                with self.assertRaisesRegex(
                    ranker.ModelInvocationError, "Malformed Gemini response"
                ) as raised:
                    await ranker.invoke_gemini(client, "gemini-test", "Question", ())

                self.assertIsInstance(raised.exception.__cause__, AttributeError)

    async def test_gemini_wraps_setup_and_api_exceptions(self):
        with patch.object(
            ranker.types,
            "GenerateContentConfig",
            side_effect=RuntimeError("invalid config"),
        ):
            with self.assertRaisesRegex(ranker.ModelInvocationError, "invalid config"):
                await ranker.invoke_gemini(MagicMock(), "gemini-test", "Question", ())

        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("connection failed")
        )
        with self.assertRaisesRegex(ranker.ModelInvocationError, "connection failed"):
            await ranker.invoke_gemini(client, "gemini-test", "Question", ())

    def test_load_api_keys_uses_google_or_gemini_key(self):
        expected = ranker.ApiKeys(
            gpt="openai-key",
            gemini="google-key",
            grok="xai-key",
            claude="anthropic-key",
        )
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "GOOGLE_API_KEY": "google-key",
                "GEMINI_API_KEY": "gemini-key",
                "XAI_API_KEY": "xai-key",
                "ANTHROPIC_API_KEY": "anthropic-key",
            },
            clear=True,
        ):
            self.assertEqual(ranker.load_api_keys(), expected)

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "openai-key",
                "GEMINI_API_KEY": "gemini-key",
                "XAI_API_KEY": "xai-key",
                "ANTHROPIC_API_KEY": "anthropic-key",
            },
            clear=True,
        ):
            self.assertEqual(
                ranker.load_api_keys(),
                ranker.ApiKeys(
                    gpt="openai-key",
                    gemini="gemini-key",
                    grok="xai-key",
                    claude="anthropic-key",
                ),
            )

    def test_load_api_keys_reports_every_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                ranker.ModelInvocationError,
                "OPENAI_API_KEY.*GOOGLE_API_KEY or GEMINI_API_KEY.*XAI_API_KEY.*ANTHROPIC_API_KEY",
            ):
                ranker.load_api_keys()


class InputParsingTests(unittest.TestCase):
    def test_read_records_accepts_an_iterable_of_lines(self):
        records = ranker.read_records(
            ["NVDA,NVIDIA Corporation\n", 'NFLX,"Netflix, Inc."\n']
        )

        self.assertEqual(
            records,
            ["NVDA,NVIDIA Corporation", 'NFLX,"Netflix, Inc."'],
        )


class PromptBuilderTests(unittest.TestCase):
    def test_build_analysis_question_for_each_criterion(self):
        for name, expected in (
            ("moat", "What moats do they have?"),
            ("power", "What pricing powers do they have?"),
        ):
            with self.subTest(criterion=name):
                question = ranker.build_analysis_question(
                    ["NVDA,NVIDIA Corporation", 'NFLX,"Netflix, Inc."'],
                    ranker.CRITERIA[name],
                )
                self.assertEqual(
                    question,
                    'NVDA,NVIDIA Corporation\nNFLX,"Netflix, Inc."\n' + expected,
                )

    def test_format_panel_responses_preserves_raw_text(self):
        prompt = ranker.format_panel_responses(
            dict(
                gpt=" GPT answer\nSecond line",
                gemini="Gemini answer",
                claude="Claude answer",
                grok="Grok answer",
            ),
        )

        self.assertEqual(
            prompt,
            """<response model="gpt">
 GPT answer
Second line
</response>
<response model="gemini">
Gemini answer
</response>
<response model="claude">
Claude answer
</response>
<response model="grok">
Grok answer
</response>""",
        )

    def test_build_ranking_question(self):
        for name, expected in (
            ("moat", "Rank them into 4 tiers by moat"),
            ("power", "Rank them into 4 tiers by pricing power"),
        ):
            with self.subTest(criterion=name):
                for batch_count in (1, 2):
                    with self.subTest(batch_count=batch_count):
                        question = ranker.build_ranking_question(
                            [
                                make_panel_responses(f"batch {i}")
                                for i in range(1, batch_count + 1)
                            ],
                            ranker.CRITERIA[name],
                        )
                        expected_contexts = []
                        for model, provider in (
                            ("gpt", "GPT"),
                            ("gemini", "Gemini"),
                            ("claude", "Claude"),
                            ("grok", "Grok"),
                        ):
                            answers = "\n".join(
                                f"{provider} batch {i}"
                                for i in range(1, batch_count + 1)
                            )
                            expected_contexts.append(
                                f'<context model="{model}">\n{answers}\n</context>'
                            )
                        self.assertEqual(
                            question, "\n".join(expected_contexts) + "\n" + expected
                        )


class ArgumentParsingTests(unittest.TestCase):
    def test_grok_priority_is_opt_in(self):
        for flags, expected in (([], False), (["--grok-priority"], True)):
            with self.subTest(flags=flags):
                with patch.object(ranker.sys, "argv", ["ranker.py", "-p", *flags]):
                    self.assertEqual(ranker.parse_args().grok_priority, expected)

    def test_p_selects_pricing_power(self):
        with patch.object(ranker.sys, "argv", ["ranker.py", "-p"]):
            self.assertEqual(ranker.parse_args().criterion, "power")

    def test_m_selects_moat(self):
        with patch.object(ranker.sys, "argv", ["ranker.py", "-m"]):
            self.assertEqual(ranker.parse_args().criterion, "moat")

    def test_exactly_one_criterion_is_required(self):
        for flags in ([], ["--grok-priority"], ["-p", "-m"], ["-m", "-p"]):
            with self.subTest(flags=flags):
                with (
                    patch.object(ranker.sys, "argv", ["ranker.py", *flags]),
                    patch.object(ranker.sys, "stderr", io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    ranker.parse_args()
                self.assertEqual(raised.exception.code, 2)

    def test_removed_flags_are_rejected(self):
        for flag, value in (
            ("--criterion", "moat"),
            ("--gpt-model", "custom"),
            ("--gemini-model", "custom"),
            ("--claude-model", "custom"),
            ("--grok-model", "custom"),
        ):
            with self.subTest(flag=flag):
                with (
                    patch.object(ranker.sys, "argv", ["ranker.py", "-p", flag, value]),
                    patch.object(ranker.sys, "stderr", io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    ranker.parse_args()
                self.assertEqual(raised.exception.code, 2)


class PanelQueryTests(unittest.IsolatedAsyncioTestCase):
    async def test_panel_query_returns_responses_in_provider_order(self):
        models = ranker.Models(
            gpt="gpt-test",
            gemini="gemini-test",
            claude="claude-test",
            grok="grok-test",
        )
        clients = make_panel_clients()
        panel = ranker.Panel(models=models, clients=clients)
        history = (("Earlier question", "Earlier answer"),)

        with (
            patch.object(
                ranker, "invoke_gpt", new=AsyncMock(return_value="GPT answer")
            ) as invoke_gpt,
            patch.object(
                ranker,
                "invoke_gemini",
                new=AsyncMock(return_value="Gemini answer"),
            ) as invoke_gemini,
            patch.object(
                ranker,
                "invoke_claude",
                new=AsyncMock(return_value="Claude answer"),
            ) as invoke_claude,
            patch.object(
                ranker,
                "invoke_grok",
                new=AsyncMock(return_value="Grok answer"),
            ) as invoke_grok,
        ):
            responses = await panel.query("Question", history)

        self.assertEqual(tuple(responses), ("gpt", "gemini", "claude", "grok"))
        self.assertEqual(
            responses,
            dict(
                gpt="GPT answer",
                gemini="Gemini answer",
                claude="Claude answer",
                grok="Grok answer",
            ),
        )
        invoke_gpt.assert_awaited_once_with(
            clients.gpt, "gpt-test", "Question", history
        )
        invoke_gemini.assert_awaited_once_with(
            clients.gemini, "gemini-test", "Question", history
        )
        invoke_claude.assert_awaited_once_with(
            clients.claude, "claude-test", "Question", history
        )
        invoke_grok.assert_awaited_once_with(
            clients.grok, "grok-test", "Question", history, priority=False
        )

    async def test_panel_passes_priority_to_grok(self):
        panel = ranker.Panel(
            models=ranker.Models("gpt-test", "gemini-test", "claude-test", "grok-test"),
            clients=make_panel_clients(),
            grok_priority=True,
        )
        with (
            patch.object(ranker, "invoke_gpt", new=AsyncMock(return_value="GPT")),
            patch.object(ranker, "invoke_gemini", new=AsyncMock(return_value="Gemini")),
            patch.object(ranker, "invoke_claude", new=AsyncMock(return_value="Claude")),
            patch.object(
                ranker, "invoke_grok", new=AsyncMock(return_value="Grok")
            ) as invoke_grok,
        ):
            await panel.query("Question")

        invoke_grok.assert_awaited_once_with(
            panel.clients.grok, "grok-test", "Question", (), priority=True
        )

    async def test_panel_query_propagates_provider_failures(self):
        models = ranker.Models(
            gpt="gpt-test",
            gemini="gemini-test",
            claude="claude-test",
            grok="grok-test",
        )
        clients = make_panel_clients()
        panel = ranker.Panel(models=models, clients=clients)
        failure = ranker.ModelInvocationError("Gemini unavailable")

        with (
            patch.object(
                ranker, "invoke_gpt", new=AsyncMock(return_value="GPT answer")
            ),
            patch.object(ranker, "invoke_gemini", new=AsyncMock(side_effect=failure)),
            patch.object(
                ranker,
                "invoke_claude",
                new=AsyncMock(return_value="Claude answer"),
            ),
            patch.object(
                ranker, "invoke_grok", new=AsyncMock(return_value="Grok answer")
            ),
        ):
            with self.assertRaisesRegex(
                ranker.ModelInvocationError, "Gemini unavailable"
            ):
                await panel.query("Question")


class BatchWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_rank_records_preserves_batch_order_and_merges_only_at_the_end(self):
        records = ["AAA,Alpha", "BBB,Beta", "CCC,Gamma", "DDD,Delta", "EEE,Echo"]
        for criterion in ranker.CRITERIA.values():
            with self.subTest(criterion=criterion.name):
                panel = ranker.Panel(
                    models=ranker.Models("gpt", "gemini", "claude", "grok"),
                    clients=make_panel_clients(),
                )
                first_question = ranker.build_analysis_question(records[:4], criterion)
                second_question = ranker.build_analysis_question(records[4:], criterion)
                second_batch_done = asyncio.Event()
                completed_batches = []
                revised = [
                    make_panel_responses("batch 1"),
                    make_panel_responses("batch 2"),
                ]
                rankings = make_panel_responses("ranking")
                ranking_question = ranker.build_ranking_question(revised, criterion)

                async def query(question, history: tuple[ranker.HistoryItem, ...] = ()):
                    if question == first_question:
                        await second_batch_done.wait()
                        return make_panel_responses("first pass 1")
                    if question == second_question:
                        return make_panel_responses("first pass 2")
                    if question == "think again":
                        if history[0][0] == second_question:
                            completed_batches.append(2)
                            second_batch_done.set()
                            return revised[1]
                        completed_batches.append(1)
                        return revised[0]
                    self.assertEqual(question, ranking_question)
                    self.assertEqual(history, ())
                    return rankings

                with (
                    patch.object(
                        ranker.Panel, "query", side_effect=query
                    ) as panel_query,
                    patch.object(
                        ranker,
                        "invoke_gpt",
                        new=AsyncMock(return_value="Final ranking"),
                    ) as merge,
                    patch.object(ranker, "progress"),
                ):
                    result = await asyncio.wait_for(
                        ranker.rank_records(records, panel, criterion), timeout=5
                    )

                self.assertEqual(result, "Final ranking")
                self.assertEqual(completed_batches, [2, 1])
                self.assertEqual(panel_query.await_count, 5)
                merge.assert_awaited_once_with(
                    panel.clients.gpt,
                    panel.models.gpt,
                    "Merge the responses into a coherent one. List any major conflicts.",
                    ((ranking_question, ranker.format_panel_responses(rankings)),),
                )

    async def test_final_ranking_and_merge_failures_propagate(self):
        for stage in ("ranking", "merge"):
            with self.subTest(stage=stage):
                failure = ranker.ModelInvocationError("Provider unavailable")
                panel = MagicMock()
                rankings = make_panel_responses("ranking")
                panel.query = AsyncMock(
                    side_effect=failure if stage == "ranking" else None,
                    return_value=rankings,
                )
                with (
                    patch.object(ranker, "progress"),
                    patch.object(
                        ranker,
                        "process_batch",
                        new=AsyncMock(return_value=make_panel_responses("batch")),
                    ),
                    patch.object(
                        ranker, "invoke_gpt", new=AsyncMock(side_effect=failure)
                    ) as merge,
                    self.assertRaisesRegex(
                        ranker.ModelInvocationError, "Provider unavailable"
                    ),
                ):
                    await ranker.rank_records(
                        ["AAA,Alpha"], panel, ranker.CRITERIA["power"]
                    )
                self.assertEqual(merge.await_count, int(stage == "merge"))

    async def test_batch_failures_prevent_ranking_and_merge(self):
        for failed_round in (1, 2):
            with self.subTest(failed_round=failed_round):
                failure = ranker.ModelInvocationError("Panel failed")
                results = (
                    [failure]
                    if failed_round == 1
                    else [make_panel_responses("first"), failure]
                )
                panel = MagicMock()
                panel.query = AsyncMock(side_effect=results)
                with (
                    patch.object(ranker, "progress"),
                    patch.object(ranker, "invoke_gpt", new=AsyncMock()) as merge,
                    self.assertRaisesRegex(ranker.ModelInvocationError, "Panel failed"),
                ):
                    await ranker.rank_records(
                        ["AAA,Alpha"], panel, ranker.CRITERIA["power"]
                    )
                self.assertEqual(panel.query.await_count, failed_round)
                merge.assert_not_awaited()

    async def test_process_batch_passes_all_first_responses_as_deeper_history(self):
        for criterion in ranker.CRITERIA.values():
            with self.subTest(criterion=criterion.name):
                await self.check_process_batch_history(criterion)

    async def check_process_batch_history(self, criterion):
        panel = MagicMock()
        first = make_panel_responses("first")
        deeper = make_panel_responses("deeper")
        panel.query = AsyncMock(side_effect=[first, deeper])
        batch = ["NVDA,NVIDIA Corporation", "NFLX,Netflix Inc."]
        initial_question = ranker.build_analysis_question(batch, criterion)

        with (
            patch.object(ranker, "progress") as progress,
        ):
            result = await ranker.process_batch(batch, 1, panel, criterion)

        self.assertEqual(result, deeper)
        progress.assert_any_call(f"Batch 1: first-pass {criterion.name} analysis")
        self.assertEqual(
            panel.query.await_args_list,
            [
                call(initial_question),
                call(
                    "think again",
                    ((initial_question, ranker.format_panel_responses(first)),),
                ),
            ],
        )

    async def test_run_uses_combined_batch_analyses_for_final_ranking(self):
        for criterion_name in ranker.CRITERIA:
            for grok_priority in (False, True):
                with self.subTest(
                    criterion=criterion_name, grok_priority=grok_priority
                ):
                    await self.check_run_ranking(criterion_name, grok_priority)

    async def check_run_ranking(self, criterion_name, grok_priority):
        flags = ["-m"] if criterion_name == "moat" else ["-p"]
        if grok_priority:
            flags.append("--grok-priority")
        with patch.object(ranker.sys, "argv", ["ranker.py", *flags]):
            args = ranker.parse_args()
        models = ranker.Models(
            gpt=ranker.MODEL_GPT_BASE,
            gemini=ranker.MODEL_GEMINI,
            claude=ranker.MODEL_CLAUDE,
            grok=ranker.MODEL_GROK_BASE,
        )
        api_keys = ranker.ApiKeys(
            gpt="openai-key",
            gemini="gemini-key",
            claude="anthropic-key",
            grok="xai-key",
        )
        gpt_client = MagicMock(name="gpt_client")
        gemini_client = MagicMock(name="gemini_client")
        claude_client = MagicMock(name="claude_client")
        grok_client = MagicMock(name="grok_client")
        gpt_context = AsyncContextManager(gpt_client)
        claude_context = AsyncContextManager(claude_client)
        grok_context = AsyncContextManager(grok_client)
        clients = ranker.PanelClients(
            gpt=gpt_client,
            gemini=gemini_client,
            claude=claude_client,
            grok=grok_client,
        )
        records = [
            "AAA,Alpha",
            "BBB,Beta",
            "CCC,Gamma",
            "DDD,Delta",
            "EEE,Echo",
        ]
        batch_finals = [
            make_panel_responses("first batch"),
            make_panel_responses("second batch"),
        ]
        query_calls = []
        rankings = make_panel_responses("ranking")

        async def query(panel, question, history=()):
            query_calls.append((panel, question, history))
            return rankings

        with (
            patch.object(ranker, "read_records", return_value=records) as read_records,
            patch.object(ranker, "load_api_keys", return_value=api_keys),
            patch.object(
                ranker,
                "AsyncOpenAI",
                return_value=gpt_context,
            ) as openai_factory,
            patch.object(
                ranker.anthropic,
                "AsyncAnthropic",
                return_value=claude_context,
            ) as claude_factory,
            patch.object(
                ranker,
                "AsyncClient",
                return_value=grok_context,
            ) as grok_factory,
            patch.object(
                ranker.genai, "Client", return_value=gemini_client
            ) as gemini_factory,
            patch.object(
                ranker, "process_batch", new=AsyncMock(side_effect=batch_finals)
            ) as process_batch,
            patch.object(ranker.Panel, "query", new=query),
            patch.object(
                ranker, "invoke_gpt", new=AsyncMock(return_value="Final ranking")
            ) as merge,
            patch.object(ranker, "progress"),
        ):
            result = await ranker.run(args)

        self.assertEqual(result, "Final ranking")
        read_records.assert_called_once_with(ranker.sys.stdin)
        self.assertEqual(
            [(question, history) for _, question, history in query_calls],
            [
                (
                    ranker.build_ranking_question(
                        batch_finals, ranker.CRITERIA[criterion_name]
                    ),
                    (),
                )
            ],
        )
        merge.assert_awaited_once_with(
            gpt_client,
            ranker.MODEL_GPT_BASE,
            ranker.MERGE_INSTRUCTION,
            ((query_calls[0][1], ranker.format_panel_responses(rankings)),),
        )
        self.assertEqual(process_batch.await_count, 2)
        self.assertCountEqual(
            [
                (arguments.args[0], arguments.args[1])
                for arguments in process_batch.await_args_list
            ],
            [(records[:4], 1), (records[4:], 2)],
        )
        panels = [arguments.args[2] for arguments in process_batch.await_args_list]
        self.assertTrue(
            all(
                arguments.args[3] == ranker.CRITERIA[criterion_name]
                for arguments in process_batch.await_args_list
            )
        )
        self.assertTrue(all(panel.models == models for panel in panels))
        self.assertTrue(all(panel.clients == clients for panel in panels))
        self.assertTrue(all(panel.grok_priority == grok_priority for panel in panels))
        self.assertTrue(all(panel is query_calls[0][0] for panel in panels))
        openai_factory.assert_called_once_with(api_key="openai-key")
        gemini_factory.assert_called_once_with(api_key="gemini-key")
        claude_factory.assert_called_once_with(api_key="anthropic-key")
        grok_factory.assert_called_once_with(api_key="xai-key")
        self.assertTrue(gpt_context.entered and gpt_context.exited)
        self.assertTrue(claude_context.entered and claude_context.exited)
        self.assertTrue(grok_context.entered and grok_context.exited)


class MainTests(unittest.TestCase):
    def test_main_reports_model_invocation_failures(self):
        args = types.SimpleNamespace()
        stdout = io.StringIO()
        stderr = io.StringIO()
        error = ranker.ModelInvocationError("panel unavailable")
        with (
            patch.object(ranker, "load_dotenv"),
            patch.object(ranker, "parse_args", return_value=args),
            patch.object(ranker, "run", new=AsyncMock(side_effect=error)) as run,
            patch.object(ranker.sys, "stdout", stdout),
            patch.object(ranker.sys, "stderr", stderr),
        ):
            exit_code = ranker.main()

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "Error: panel unavailable\n")
        run.assert_awaited_once_with(args)


if __name__ == "__main__":
    unittest.main()
