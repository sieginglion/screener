import io
import os
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

import moat_ranker


class AsyncContextManager:
    def __init__(self, value):
        self.value = value
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        self.exited = True
        return False


def make_panel_clients():
    return moat_ranker.PanelClients(
        gpt=MagicMock(name="gpt_client"),
        gemini=MagicMock(name="gemini_client"),
        claude=MagicMock(name="claude_client"),
        grok=MagicMock(name="grok_client"),
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

        result = await moat_ranker.invoke_gpt(client, "gpt-test", "Question", ())

        self.assertEqual(result, "GPT result")

    async def test_gemini_returns_response_text(self):
        response = types.SimpleNamespace(
            prompt_feedback=None,
            candidates=[
                types.SimpleNamespace(
                    finish_reason=moat_ranker.types.FinishReason.STOP,
                    finish_message=None,
                )
            ],
            text="Gemini result",
        )
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(return_value=response)

        result = await moat_ranker.invoke_gemini(
            client, "gemini-test", "Question", ()
        )

        self.assertEqual(result, "Gemini result")

    async def test_grok_returns_stripped_response_text(self):
        response = types.SimpleNamespace(
            error=None,
            finish_reason="REASON_STOP",
            content="  Grok result  ",
        )
        chat = MagicMock()
        chat.sample = AsyncMock(return_value=response)
        client = MagicMock()
        client.chat.create.return_value = chat

        result = await moat_ranker.invoke_grok(client, "grok-test", "Question", ())

        self.assertEqual(result, "Grok result")

    async def test_grok_rejects_api_and_response_failures(self):
        responses = [
            (
                types.SimpleNamespace(
                    error=types.SimpleNamespace(
                        code="server_error", message="Model failure"
                    ),
                    finish_reason="REASON_STOP",
                    content="Grok result",
                ),
                "Model failure",
            ),
            (
                types.SimpleNamespace(
                    error=None,
                    finish_reason="REASON_MAX_LEN",
                    content="Partial Grok result",
                ),
                "stopped with 'REASON_MAX_LEN'",
            ),
            (
                types.SimpleNamespace(
                    error=None,
                    finish_reason="REASON_STOP",
                    content=None,
                ),
                "contained no text",
            ),
            (
                types.SimpleNamespace(
                    error=None,
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
                    moat_ranker.ModelInvocationError, expected_error
                ):
                    await moat_ranker.invoke_grok(client, "grok-test", "Question", ())

    async def test_grok_wraps_setup_and_api_exceptions(self):
        client = MagicMock()
        client.chat.create.side_effect = RuntimeError("invalid model")
        with self.assertRaisesRegex(
            moat_ranker.ModelInvocationError, "invalid model"
        ):
            await moat_ranker.invoke_grok(client, "grok-test", "Question", ())

        chat = MagicMock()
        chat.sample = AsyncMock(side_effect=RuntimeError("connection failed"))
        client = MagicMock()
        client.chat.create.return_value = chat
        with self.assertRaisesRegex(
            moat_ranker.ModelInvocationError, "connection failed"
        ):
            await moat_ranker.invoke_grok(client, "grok-test", "Question", ())

    async def test_claude_returns_response_text(self):
        response = types.SimpleNamespace(
            stop_reason="end_turn",
            content=[types.SimpleNamespace(type="text", text="Claude result")],
        )
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=response)

        result = await moat_ranker.invoke_claude(
            client, "claude-test", "Question", ()
        )

        self.assertEqual(result, "Claude result")

    async def test_gemini_rejects_api_and_response_failures(self):
        responses = [
            (
                types.SimpleNamespace(
                    prompt_feedback=types.SimpleNamespace(
                        block_reason=moat_ranker.types.BlockedReason.SAFETY
                    ),
                    candidates=None,
                    text=None,
                ),
                "prompt was blocked",
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
                            finish_reason=moat_ranker.types.FinishReason.MAX_TOKENS,
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
                            finish_reason=moat_ranker.types.FinishReason.STOP,
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
                            finish_reason=moat_ranker.types.FinishReason.STOP,
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
                    moat_ranker.ModelInvocationError, expected_error
                ):
                    await moat_ranker.invoke_gemini(
                        client, "gemini-test", "Question", ()
                    )

    async def test_gemini_wraps_setup_and_api_exceptions(self):
        with patch.object(
            moat_ranker.types,
            "GenerateContentConfig",
            side_effect=RuntimeError("invalid config"),
        ):
            with self.assertRaisesRegex(
                moat_ranker.ModelInvocationError, "invalid config"
            ):
                await moat_ranker.invoke_gemini(
                    MagicMock(), "gemini-test", "Question", ()
                )

        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(
            side_effect=RuntimeError("connection failed")
        )
        with self.assertRaisesRegex(
            moat_ranker.ModelInvocationError, "connection failed"
        ):
            await moat_ranker.invoke_gemini(client, "gemini-test", "Question", ())

    def test_load_api_keys_uses_google_or_gemini_key(self):
        expected = moat_ranker.ApiKeys(
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
            self.assertEqual(moat_ranker.load_api_keys(), expected)

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
                moat_ranker.load_api_keys(),
                moat_ranker.ApiKeys(
                    gpt="openai-key",
                    gemini="gemini-key",
                    grok="xai-key",
                    claude="anthropic-key",
                ),
            )

    def test_load_api_keys_reports_every_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                moat_ranker.ModelInvocationError,
                "OPENAI_API_KEY.*GOOGLE_API_KEY or GEMINI_API_KEY.*XAI_API_KEY.*ANTHROPIC_API_KEY",
            ):
                moat_ranker.load_api_keys()


class PromptBuilderTests(unittest.TestCase):
    def test_build_moat_question(self):
        question = moat_ranker.build_moat_question(
            ["NVDA,NVIDIA Corporation", 'NFLX,"Netflix, Inc."']
        )

        self.assertEqual(
            question,
            'NVDA,NVIDIA Corporation\nNFLX,"Netflix, Inc."\nWhat are their moats?',
        )

    def test_build_synthesis_prompt(self):
        prompt = moat_ranker.build_synthesis_prompt(
            "Question",
            moat_ranker.PanelResponses(
                gpt="GPT answer",
                gemini="Gemini answer",
                claude="Claude answer",
                grok="Grok answer",
            ),
        )

        self.assertEqual(
            prompt,
            '''<prompt>
Question
</prompt>
<response model="gpt">
GPT answer
</response>
<response model="gemini">
Gemini answer
</response>
<response model="claude">
Claude answer
</response>
<response model="grok">
Grok answer
</response>
Merge the responses into a coherent one. List any major conflicts.
''',
        )

    def test_build_ranking_question(self):
        question = moat_ranker.build_ranking_question(
            ["First batch analysis", "Second batch analysis"]
        )

        self.assertEqual(
            question,
            "<context>\nFirst batch analysis\nSecond batch analysis\n</context>\n"
            "Rank them into 4 tiers by moat width",
        )


class PanelSynthesisTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_panel_returns_responses_in_provider_order(self):
        models = moat_ranker.Models(
            gpt="gpt-test",
            gemini="gemini-test",
            claude="claude-test",
            grok="grok-test",
        )
        clients = make_panel_clients()
        history = (("Earlier question", "Earlier answer"),)

        with (
            patch.object(
                moat_ranker, "invoke_gpt", new=AsyncMock(return_value="GPT answer")
            ) as invoke_gpt,
            patch.object(
                moat_ranker,
                "invoke_gemini",
                new=AsyncMock(return_value="Gemini answer"),
            ) as invoke_gemini,
            patch.object(
                moat_ranker,
                "invoke_claude",
                new=AsyncMock(return_value="Claude answer"),
            ) as invoke_claude,
            patch.object(
                moat_ranker,
                "invoke_grok",
                new=AsyncMock(return_value="Grok answer"),
            ) as invoke_grok,
        ):
            responses = await moat_ranker.query_panel(
                "Question", models, clients, history
            )

        self.assertEqual(
            responses,
            moat_ranker.PanelResponses(
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
            clients.grok, "grok-test", "Question", history
        )

    async def test_query_panel_propagates_provider_failures(self):
        models = moat_ranker.Models(
            gpt="gpt-test",
            gemini="gemini-test",
            claude="claude-test",
            grok="grok-test",
        )
        clients = make_panel_clients()
        failure = moat_ranker.ModelInvocationError("Gemini unavailable")

        with (
            patch.object(
                moat_ranker, "invoke_gpt", new=AsyncMock(return_value="GPT answer")
            ),
            patch.object(
                moat_ranker, "invoke_gemini", new=AsyncMock(side_effect=failure)
            ),
            patch.object(
                moat_ranker,
                "invoke_claude",
                new=AsyncMock(return_value="Claude answer"),
            ),
            patch.object(
                moat_ranker, "invoke_grok", new=AsyncMock(return_value="Grok answer")
            ),
        ):
            with self.assertRaisesRegex(
                moat_ranker.ModelInvocationError, "Gemini unavailable"
            ):
                await moat_ranker.query_panel("Question", models, clients)

    async def test_deliberate_does_not_synthesize_after_panel_failure(self):
        models = moat_ranker.Models(
            gpt="gpt-test",
            gemini="gemini-test",
            claude="claude-test",
            grok="grok-test",
        )
        clients = make_panel_clients()

        with (
            patch.object(
                moat_ranker,
                "query_panel",
                new=AsyncMock(
                    side_effect=moat_ranker.ModelInvocationError("Grok unavailable")
                ),
            ),
            patch.object(moat_ranker, "synthesize", new=AsyncMock()) as synthesize,
        ):
            with self.assertRaisesRegex(
                moat_ranker.ModelInvocationError, "Grok unavailable"
            ):
                await moat_ranker.deliberate("Question", models, clients)

        synthesize.assert_not_awaited()

    async def test_deliberate_synthesizes_full_provider_responses(self):
        models = moat_ranker.Models(
            gpt="gpt-test",
            gemini="gemini-test",
            claude="claude-test",
            grok="grok-test",
        )
        clients = make_panel_clients()
        history = (("Earlier question", "Earlier answer"),)

        with (
            patch.object(
                moat_ranker,
                "invoke_gpt",
                new=AsyncMock(side_effect=["GPT panel answer", "Merged answer"]),
            ) as invoke_gpt,
            patch.object(
                moat_ranker,
                "invoke_gemini",
                new=AsyncMock(return_value="Gemini panel answer"),
            ),
            patch.object(
                moat_ranker,
                "invoke_claude",
                new=AsyncMock(return_value="Claude panel answer"),
            ),
            patch.object(
                moat_ranker,
                "invoke_grok",
                new=AsyncMock(return_value="Grok panel answer"),
            ),
        ):
            result = await moat_ranker.deliberate("Question", models, clients, history)

        self.assertEqual(result, "Merged answer")
        synthesis_prompt = invoke_gpt.await_args_list[1].args[2]
        self.assertEqual(
            synthesis_prompt,
            moat_ranker.build_synthesis_prompt(
                "Question",
                moat_ranker.PanelResponses(
                    gpt="GPT panel answer",
                    gemini="Gemini panel answer",
                    claude="Claude panel answer",
                    grok="Grok panel answer",
                ),
            ),
        )
        self.assertEqual(invoke_gpt.await_args_list[1].args[3], history)


class BatchWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_batch_passes_first_synthesis_as_deeper_history(self):
        models = moat_ranker.Models(
            gpt="gpt-test",
            gemini="gemini-test",
            claude="claude-test",
            grok="grok-test",
        )
        clients = make_panel_clients()
        batch = ["NVDA,NVIDIA Corporation", "NFLX,Netflix Inc."]
        initial_question = moat_ranker.build_moat_question(batch)

        with (
            patch.object(
                moat_ranker,
                "deliberate",
                new=AsyncMock(side_effect=["First synthesis", "Deeper synthesis"]),
            ) as deliberate,
            patch.object(moat_ranker, "progress"),
        ):
            result = await moat_ranker.process_batch(batch, 1, models, clients)

        self.assertEqual(result, "Deeper synthesis")
        self.assertEqual(
            deliberate.await_args_list,
            [
                call(initial_question, models, clients),
                call(
                    "think deeper",
                    models,
                    clients,
                    ((initial_question, "First synthesis"),),
                ),
            ],
        )

    async def test_run_uses_combined_batch_analyses_for_final_ranking(self):
        args = types.SimpleNamespace(
            batch_size=4,
            gpt_model="gpt-test",
            gemini_model="gemini-test",
            grok_model="grok-test",
            claude_model="claude-test",
        )
        models = moat_ranker.Models(
            gpt="gpt-test",
            gemini="gemini-test",
            claude="claude-test",
            grok="grok-test",
        )
        api_keys = moat_ranker.ApiKeys(
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
        clients = moat_ranker.PanelClients(
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
        batch_finals = ["First batch analysis", "Second batch analysis"]

        with (
            patch.object(moat_ranker, "read_records", return_value=records),
            patch.object(moat_ranker, "load_api_keys", return_value=api_keys),
            patch.object(
                moat_ranker,
                "AsyncOpenAI",
                return_value=gpt_context,
            ) as openai_factory,
            patch.object(
                moat_ranker.anthropic,
                "AsyncAnthropic",
                return_value=claude_context,
            ) as claude_factory,
            patch.object(
                moat_ranker,
                "AsyncClient",
                return_value=grok_context,
            ) as grok_factory,
            patch.object(
                moat_ranker.genai, "Client", return_value=gemini_client
            ) as gemini_factory,
            patch.object(
                moat_ranker, "process_batch", new=AsyncMock(side_effect=batch_finals)
            ) as process_batch,
            patch.object(
                moat_ranker, "deliberate", new=AsyncMock(return_value="Final ranking")
            ) as deliberate,
            patch.object(moat_ranker, "progress"),
        ):
            result = await moat_ranker.run(args)

        self.assertEqual(result, "Final ranking")
        deliberate.assert_awaited_once_with(
            moat_ranker.build_ranking_question(batch_finals), models, clients
        )
        process_batch.assert_has_awaits(
            [
                call(records[:4], 1, models, clients),
                call(records[4:], 2, models, clients),
            ],
            any_order=True,
        )
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
        error = moat_ranker.ModelInvocationError("panel unavailable")
        with (
            patch.object(moat_ranker, "load_dotenv"),
            patch.object(moat_ranker, "parse_args", return_value=args),
            patch.object(moat_ranker, "run", new=AsyncMock(side_effect=error)) as run,
            patch.object(moat_ranker.sys, "stdout", stdout),
            patch.object(moat_ranker.sys, "stderr", stderr),
        ):
            exit_code = moat_ranker.main()

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "Error: panel unavailable\n")
        run.assert_awaited_once_with(args)


if __name__ == "__main__":
    unittest.main()
