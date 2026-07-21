import os
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, call, patch

import moat_ranker


class AsyncContextManager:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class ProviderInvocationTests(unittest.IsolatedAsyncioTestCase):
    async def test_gpt_returns_response_text(self):
        response = types.SimpleNamespace(
            error=None,
            status="completed",
            output_text="GPT result",
        )
        client = MagicMock()
        client.responses.create = AsyncMock(return_value=response)

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
            patch.object(
                moat_ranker,
                "AsyncOpenAI",
                return_value=AsyncContextManager(client),
            ),
        ):
            result = await moat_ranker.invoke_gpt("gpt-test", "Question", ())

        self.assertEqual(result, "GPT result")

    async def test_gemini_returns_response_text(self):
        response = types.SimpleNamespace(text="Gemini result")
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(return_value=response)

        with (
            patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}),
            patch.object(moat_ranker.genai, "Client", return_value=client),
        ):
            result = await moat_ranker.invoke_gemini("gemini-test", "Question", ())

        self.assertEqual(result, "Gemini result")

    async def test_grok_returns_stripped_response_text(self):
        response = types.SimpleNamespace(content="  Grok result  ")
        chat = MagicMock()
        chat.sample = AsyncMock(return_value=response)
        client = MagicMock()
        client.chat.create.return_value = chat

        with (
            patch.dict(os.environ, {"XAI_API_KEY": "test-key"}),
            patch.object(
                moat_ranker,
                "AsyncClient",
                return_value=AsyncContextManager(client),
            ),
        ):
            result = await moat_ranker.invoke_grok("grok-test", "Question", ())

        self.assertEqual(result, "Grok result")

    async def test_claude_returns_response_text(self):
        response = types.SimpleNamespace(
            stop_reason="end_turn",
            content=[types.SimpleNamespace(type="text", text="Claude result")],
        )
        client = MagicMock()
        client.messages.create = AsyncMock(return_value=response)

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}),
            patch.object(
                moat_ranker.anthropic,
                "AsyncAnthropic",
                return_value=AsyncContextManager(client),
            ),
        ):
            result = await moat_ranker.invoke_claude("claude-test", "Question", ())

        self.assertEqual(result, "Claude result")

    async def test_missing_keys_preserve_provider_failure_policy(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(moat_ranker.GPTInvocationError):
                await moat_ranker.invoke_gpt("gpt-test", "Question", ())
            with self.assertRaises(moat_ranker.ClaudeInvocationError):
                await moat_ranker.invoke_claude("claude-test", "Question", ())

            self.assertEqual(
                await moat_ranker.invoke_gemini("gemini-test", "Question", ()),
                "Error: GEMINI_API_KEY/GOOGLE_API_KEY not found.",
            )
            self.assertEqual(
                await moat_ranker.invoke_grok("grok-test", "Question", ()),
                "Error: XAI_API_KEY not found.",
            )


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
            ("GPT answer", "Gemini answer", "Claude answer", "Grok answer"),
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
            grok="grok-test",
            claude="claude-test",
        )
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
            responses = await moat_ranker.query_panel("Question", models, history)

        self.assertEqual(
            responses,
            ("GPT answer", "Gemini answer", "Claude answer", "Grok answer"),
        )
        invoke_gpt.assert_awaited_once_with("gpt-test", "Question", history)
        invoke_gemini.assert_awaited_once_with("gemini-test", "Question", history)
        invoke_claude.assert_awaited_once_with("claude-test", "Question", history)
        invoke_grok.assert_awaited_once_with("grok-test", "Question", history)

    async def test_deliberate_synthesizes_full_provider_responses(self):
        models = moat_ranker.Models(
            gpt="gpt-test",
            gemini="gemini-test",
            grok="grok-test",
            claude="claude-test",
        )
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
            result = await moat_ranker.deliberate("Question", models, history)

        self.assertEqual(result, "Merged answer")
        synthesis_prompt = invoke_gpt.await_args_list[1].args[1]
        self.assertEqual(
            synthesis_prompt,
            moat_ranker.build_synthesis_prompt(
                "Question",
                (
                    "GPT panel answer",
                    "Gemini panel answer",
                    "Claude panel answer",
                    "Grok panel answer",
                ),
            ),
        )
        self.assertEqual(invoke_gpt.await_args_list[1].args[2], history)


class BatchWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_batch_passes_first_synthesis_as_deeper_history(self):
        models = moat_ranker.Models("gpt-test", "gemini-test", "grok-test", "claude-test")
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
            result = await moat_ranker.process_batch(batch, 1, models)

        self.assertEqual(result, "Deeper synthesis")
        self.assertEqual(
            deliberate.await_args_list,
            [
                call(initial_question, models),
                call("think deeper", models, ((initial_question, "First synthesis"),)),
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
        models = moat_ranker.Models("gpt-test", "gemini-test", "grok-test", "claude-test")
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
            patch.object(
                moat_ranker, "process_batch", new=AsyncMock(side_effect=batch_finals)
            ),
            patch.object(
                moat_ranker, "deliberate", new=AsyncMock(return_value="Final ranking")
            ) as deliberate,
            patch.object(moat_ranker, "progress"),
        ):
            result = await moat_ranker.run(args)

        self.assertEqual(result, "Final ranking")
        deliberate.assert_awaited_once_with(
            moat_ranker.build_ranking_question(batch_finals), models
        )


if __name__ == "__main__":
    unittest.main()
