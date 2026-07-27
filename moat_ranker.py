#!/usr/bin/env python3
"""Rank the moat width of CSV-style stock lists using the Pareja model panel.

Read one security per line from standard input, for example:

    NVDA,NVIDIA Corporation
    PLTR,Palantir Technologies Inc.
    NFLX,"Netflix, Inc."

The script processes four securities at a time.  For each batch it asks the
four Pareja models about their moats, has GPT synthesize their answers, asks
each model to "think deeper" with that synthesis as shared history, and
synthesizes again.  It then asks all four models to rank the combined batch
findings into four moat-width tiers and performs one final GPT synthesis.

Failure behavior: fail as soon as a provider error is propagated. Do not
synthesize a partial panel or emit a final ranking after a failed deliberation.

Retry behavior: use each provider SDK's default retry policy; do not add
application-level retries.  In particular, leave OpenAI's ``max_retries``
unset and do not override xAI's default gRPC channel options.

Required environment variables:
  OPENAI_API_KEY, GOOGLE_API_KEY (or GEMINI_API_KEY), XAI_API_KEY,
  ANTHROPIC_API_KEY

Install dependencies with:
  uv sync
"""

import argparse
import asyncio
import csv
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Sequence, TypeAlias

import anthropic
from anthropic.types import MessageParam
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from openai.types.responses import ResponseInputItemParam
from xai_sdk import AsyncClient
from xai_sdk.chat import assistant, user

# These are the model variants used by pareja.py.
MODEL_GPT_BASE = "gpt-5.6-terra"
MODEL_GEMINI = "gemini-3.6-flash"
MODEL_GROK_BASE = "grok-4.5"
MODEL_CLAUDE = "claude-opus-5"
BATCH_SIZE = 4
MAX_CONCURRENT_BATCHES = 4

HistoryItem: TypeAlias = tuple[str, str]


class ModelInvocationError(RuntimeError):
    """Raised when a model cannot provide a usable response."""


@dataclass(frozen=True)
class Models:
    gpt: str
    gemini: str
    claude: str
    grok: str


@dataclass(frozen=True)
class ApiKeys:
    gpt: str
    gemini: str
    claude: str
    grok: str


@dataclass(frozen=True)
class PanelClients:
    gpt: AsyncOpenAI
    gemini: genai.Client
    claude: anthropic.AsyncAnthropic
    grok: AsyncClient


@dataclass(frozen=True)
class PanelResponses:
    gpt: str
    gemini: str
    claude: str
    grok: str

    def items(self) -> tuple[tuple[str, str], ...]:
        """Return responses in the panel's canonical provider order."""
        return (
            ("gpt", self.gpt),
            ("gemini", self.gemini),
            ("claude", self.claude),
            ("grok", self.grok),
        )


@dataclass(frozen=True)
class Panel:
    """The models and clients that make up the deliberation panel."""

    models: Models
    clients: PanelClients

    async def query(
        self, question: str, history: Sequence[HistoryItem] = ()
    ) -> PanelResponses:
        """Ask the full model panel concurrently."""
        gpt, gemini, claude, grok = await asyncio.gather(
            invoke_gpt(self.clients.gpt, self.models.gpt, question, history),
            invoke_gemini(self.clients.gemini, self.models.gemini, question, history),
            invoke_claude(self.clients.claude, self.models.claude, question, history),
            invoke_grok(self.clients.grok, self.models.grok, question, history),
        )
        return PanelResponses(gpt=gpt, gemini=gemini, claude=claude, grok=grok)

    async def deliberate(
        self, question: str, history: Sequence[HistoryItem] = ()
    ) -> str:
        """Ask the panel, then have GPT merge its responses."""
        responses = await self.query(question, history)
        prompt = build_synthesis_prompt(question, responses)
        return await invoke_gpt(self.clients.gpt, self.models.gpt, prompt, history)


def progress(message: str) -> None:
    """Keep stdout reserved for the final synthesis."""
    print(message, file=sys.stderr, flush=True)


def require_text(provider: str, text: object) -> str:
    """Return a nonblank response or raise a consistent provider error."""
    if not isinstance(text, str) or not text.strip():
        raise ModelInvocationError(f"{provider} response contained no text.")
    return text


async def invoke_gpt(
    client: AsyncOpenAI, model: str, question: str, history: Sequence[HistoryItem]
) -> str:
    messages: list[ResponseInputItemParam] = []
    for prior_question, prior_answer in history:
        messages.append({"role": "user", "content": prior_question})
        messages.append({"role": "assistant", "content": prior_answer})
    messages.append({"role": "user", "content": question})

    try:
        response = await client.responses.create(
            model=model,
            input=messages,
            reasoning={"effort": "xhigh", "mode": "standard"},
        )
        response_status = response.status
        output_text = response.output_text
    except Exception as exc:
        raise ModelInvocationError(f"Error invoking OpenAI API: {exc}") from exc

    if response_status != "completed":
        raise ModelInvocationError(f"GPT response status {response_status!r}.")

    return require_text("GPT", output_text)


async def invoke_gemini(
    client: genai.Client, model: str, question: str, history: Sequence[HistoryItem]
) -> str:
    try:
        contents = []
        for prior_question, prior_answer in history:
            contents.append(
                types.Content(role="user", parts=[types.Part(text=prior_question)])
            )
            contents.append(
                types.Content(role="model", parts=[types.Part(text=prior_answer)])
            )
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH
            )
        )
        response = await client.aio.models.generate_content(
            model=model, contents=contents, config=config
        )

    except Exception as exc:
        raise ModelInvocationError(f"Error invoking Gemini API: {exc}") from exc

    try:
        candidates = response.candidates
        if not candidates:
            raise ModelInvocationError("Gemini response contained no candidates.")

        candidate = candidates[0]
        finish_reason = candidate.finish_reason
        if finish_reason != types.FinishReason.STOP:
            reason = (
                finish_reason.value if finish_reason is not None else "unknown reason"
            )
            detail = candidate.finish_message
            detail_text = f": {detail}" if detail else ""
            raise ModelInvocationError(
                f"Gemini response stopped with {reason!r}{detail_text}"
            )

        output_text = response.text
    except (AttributeError, IndexError, TypeError) as exc:
        raise ModelInvocationError(f"Malformed Gemini response: {exc}") from exc

    return require_text("Gemini", output_text)


async def invoke_grok(
    client: AsyncClient, model: str, question: str, history: Sequence[HistoryItem]
) -> str:
    try:
        chat = client.chat.create(model=model, tools=[], reasoning_effort="high")
        for prior_question, prior_answer in history:
            chat.append(user(prior_question))
            chat.append(assistant(prior_answer))
        chat.append(user(question))
        response = await chat.sample()
    except Exception as exc:
        raise ModelInvocationError(f"Error invoking xAI API: {exc}") from exc

    try:
        finish_reason = response.finish_reason
        content = response.content
    except AttributeError as exc:
        raise ModelInvocationError(f"Malformed Grok response: {exc}") from exc

    if finish_reason != "REASON_STOP":
        raise ModelInvocationError(f"Grok response stopped with {finish_reason!r}.")

    return require_text("Grok", content).strip()


async def invoke_claude(
    client: anthropic.AsyncAnthropic,
    model: str,
    question: str,
    history: Sequence[HistoryItem],
) -> str:
    messages: list[MessageParam] = []
    for prior_question, prior_answer in history:
        messages.append({"role": "user", "content": prior_question})
        messages.append({"role": "assistant", "content": prior_answer})
    messages.append({"role": "user", "content": question})

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=65536,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "xhigh"},
        )
    except Exception as exc:
        raise ModelInvocationError(f"Error invoking Anthropic API: {exc}") from exc

    if response.stop_reason != "end_turn":
        raise ModelInvocationError(
            f"Claude response stopped with {response.stop_reason!r}."
        )
    text = "\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    return require_text("Claude", text)


def build_moat_question(batch: Sequence[str]) -> str:
    return "\n".join(batch) + "\nWhat are their moats?"


def build_synthesis_prompt(question: str, responses: PanelResponses) -> str:
    responses_text = "\n".join(
        f'<response model="{model}">\n{text}\n</response>'
        for model, text in responses.items()
    )
    return f"""<prompt>
{question}
</prompt>
{responses_text}
Merge the responses into a coherent one. List any major conflicts.
"""


def build_ranking_question(batch_finals: Sequence[str]) -> str:
    return (
        "<context>\n"
        + "\n".join(batch_finals)
        + "\n</context>\nRank them into 4 tiers by moat width"
    )


async def process_batch(batch: Sequence[str], batch_number: int, panel: Panel) -> str:
    initial_question = build_moat_question(batch)
    progress(f"Batch {batch_number}: first-pass moat analysis")
    first_synthesis = await panel.deliberate(initial_question)

    # In pareja.py, the next turn's history contains the prior user question and
    # GPT's displayed synthesis. Give every model that same shared history.
    # The requested next user turn is deliberately just these two words.
    deeper_history = ((initial_question, first_synthesis),)
    progress(f"Batch {batch_number}: deeper analysis")
    return await panel.deliberate("think deeper", deeper_history)


async def rank_records(records: Sequence[str], panel: Panel) -> str:
    """Process batches and produce the final panel ranking."""
    batches = [
        records[index : index + BATCH_SIZE]
        for index in range(0, len(records), BATCH_SIZE)
    ]
    batch_semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

    async def process_batch_with_limit(batch_number: int, batch: Sequence[str]) -> str:
        async with batch_semaphore:
            return await process_batch(batch, batch_number, panel)

    batch_finals = await asyncio.gather(
        *(
            process_batch_with_limit(batch_number, batch)
            for batch_number, batch in enumerate(batches, start=1)
        )
    )

    ranking_question = build_ranking_question(batch_finals)
    progress("Final ranking: sending all batch findings to the model panel")
    return await panel.deliberate(ranking_question)


def read_records(lines: Iterable[str]) -> list[str]:
    records = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        try:
            fields = next(csv.reader([line], strict=True))
        except csv.Error as exc:
            raise ValueError(f"Invalid CSV on input line {line_number}: {exc}") from exc
        if len(fields) < 2 or not fields[0].strip() or not fields[1].strip():
            raise ValueError(
                f"Input line {line_number} must contain a ticker and company name."
            )
        # Preserve the supplied CSV spelling (including quoted company names) in prompts.
        records.append(line)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank stock moats from CSV records supplied on standard input."
    )
    parser.add_argument("--gpt-model", default=MODEL_GPT_BASE)
    parser.add_argument("--gemini-model", default=MODEL_GEMINI)
    parser.add_argument("--claude-model", default=MODEL_CLAUDE)
    parser.add_argument("--grok-model", default=MODEL_GROK_BASE)
    return parser.parse_args()


def load_api_keys() -> ApiKeys:
    gpt = os.getenv("OPENAI_API_KEY", "").strip()
    gemini = (
        os.getenv("GOOGLE_API_KEY", "").strip()
        or os.getenv("GEMINI_API_KEY", "").strip()
    )
    grok = os.getenv("XAI_API_KEY", "").strip()
    claude = os.getenv("ANTHROPIC_API_KEY", "").strip()

    missing = []
    if not gpt:
        missing.append("OPENAI_API_KEY")
    if not gemini:
        missing.append("GOOGLE_API_KEY or GEMINI_API_KEY")
    if not grok:
        missing.append("XAI_API_KEY")
    if not claude:
        missing.append("ANTHROPIC_API_KEY")
    if missing:
        raise ModelInvocationError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    return ApiKeys(gpt=gpt, gemini=gemini, claude=claude, grok=grok)


async def run(args: argparse.Namespace) -> str:
    records = read_records(sys.stdin)
    if not records:
        raise ValueError("No non-empty CSV records were supplied on standard input.")

    models = Models(
        gpt=args.gpt_model,
        gemini=args.gemini_model,
        claude=args.claude_model,
        grok=args.grok_model,
    )
    api_keys = load_api_keys()

    async with (
        AsyncOpenAI(api_key=api_keys.gpt) as gpt_client,
        anthropic.AsyncAnthropic(api_key=api_keys.claude) as claude_client,
        AsyncClient(api_key=api_keys.grok) as grok_client,
    ):
        clients = PanelClients(
            gpt=gpt_client,
            gemini=genai.Client(api_key=api_keys.gemini),
            claude=claude_client,
            grok=grok_client,
        )
        return await rank_records(records, Panel(models=models, clients=clients))


def main() -> int:
    load_dotenv()
    args = parse_args()
    try:
        print(asyncio.run(run(args)))
    except (
        ValueError,
        csv.Error,
        ModelInvocationError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
