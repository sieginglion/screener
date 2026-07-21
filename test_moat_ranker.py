import os
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

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


class PanelSynthesisTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_llm_synthesizes_full_provider_responses(self):
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
            result = await moat_ranker.call_llm("Question", models, history)

        self.assertEqual(result, "Merged answer")
        synthesis_prompt = invoke_gpt.await_args_list[1].args[1]
        self.assertIn("GPT panel answer", synthesis_prompt)
        self.assertIn("Gemini panel answer", synthesis_prompt)
        self.assertIn("Claude panel answer", synthesis_prompt)
        self.assertIn("Grok panel answer", synthesis_prompt)
        self.assertEqual(invoke_gpt.await_args_list[1].args[2], history)


if __name__ == "__main__":
    unittest.main()
