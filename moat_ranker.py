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

Retry behavior: leave ``max_retries`` unset so the OpenAI SDK retries transient
failures twice using its default policy.

Required environment variables:
  OPENAI_API_KEY, GOOGLE_API_KEY (or GEMINI_API_KEY), XAI_API_KEY,
  ANTHROPIC_API_KEY

Install dependencies with:
  pip install anthropic google-genai openai python-dotenv xai-sdk
"""

import argparse
import asyncio
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from xai_sdk import AsyncClient
from xai_sdk.chat import assistant, user

# These are the model variants used by pareja.py.
MODEL_GPT_BASE = "gpt-5.6-terra"
MODEL_GEMINI = "gemini-3.6-flash"
MODEL_GROK_BASE = "grok-4.5"
MODEL_CLAUDE = "claude-opus-4-8"

HistoryItem = Tuple[str, str]


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
    grok: str
    claude: str


@dataclass(frozen=True)
class PanelClients:
    gpt: AsyncOpenAI
    gemini: genai.Client
    grok: AsyncClient
    claude: anthropic.AsyncAnthropic


@dataclass(frozen=True)
class PanelResponses:
    gpt: str
    gemini: str
    claude: str
    grok: str


def progress(message: str) -> None:
    """Keep stdout reserved for the final synthesis."""
    print(message, file=sys.stderr, flush=True)


async def invoke_gpt(
    client: AsyncOpenAI, model: str, question: str, history: Sequence[HistoryItem]
) -> str:
    messages: List[Dict[str, str]] = []
    for prior_question, prior_answer in history:
        messages.append({"role": "user", "content": prior_question})
        messages.append({"role": "assistant", "content": prior_answer})
    messages.append({"role": "user", "content": question})

    try:
        response = await client.responses.create(
            model=model,
            input=messages,
            reasoning={"effort": "xhigh"},
        )
        response_status = response.status
        output_text = response.output_text
        output_is_blank = not output_text.strip()
    except Exception as exc:
        raise ModelInvocationError(f"Error invoking OpenAI API: {exc}") from exc

    if response_status != "completed":
        raise ModelInvocationError(f"GPT response status {response_status!r}.")

    if output_is_blank:
        raise ModelInvocationError("GPT response contained no text.")

    return output_text


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
        if not output_text or not output_text.strip():
            raise ModelInvocationError("Gemini response contained no text.")
        return output_text
    except ModelInvocationError:
        raise
    except Exception as exc:
        raise ModelInvocationError(f"Error invoking Gemini API: {exc}") from exc


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

        finish_reason = response.finish_reason
        if finish_reason != "REASON_STOP":
            raise ModelInvocationError(f"Grok response stopped with {finish_reason!r}.")

        content = response.content
        if not content.strip():
            raise ModelInvocationError("Grok response contained no text.")
        return content.strip()
    except ModelInvocationError:
        raise
    except Exception as exc:
        raise ModelInvocationError(f"Error invoking xAI API: {exc}") from exc


async def invoke_claude(
    client: anthropic.AsyncAnthropic,
    model: str,
    question: str,
    history: Sequence[HistoryItem],
) -> str:
    messages = []
    for prior_question, prior_answer in history:
        messages.append({"role": "user", "content": prior_question})
        messages.append({"role": "assistant", "content": prior_answer})
    messages.append({"role": "user", "content": question})

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=16384,
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": "xhigh"},
        )
        if response.stop_reason != "end_turn":
            raise ModelInvocationError(
                f"Claude response stopped with {response.stop_reason!r}."
            )
        text = "\n".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not text:
            raise ModelInvocationError("Claude response contained no text.")
        return text
    except ModelInvocationError:
        raise
    except Exception as exc:
        raise ModelInvocationError(f"Error invoking Anthropic API: {exc}") from exc


def build_moat_question(batch: Sequence[str]) -> str:
    return "\n".join(batch) + "\nWhat are their moats?"


def build_synthesis_prompt(question: str, responses: PanelResponses) -> str:
    response_blocks = [
        f'''<response model="gpt">
{responses.gpt}
</response>''',
        f'''<response model="gemini">
{responses.gemini}
</response>''',
        f'''<response model="claude">
{responses.claude}
</response>''',
        f'''<response model="grok">
{responses.grok}
</response>''',
    ]
    return f'''<prompt>
{question}
</prompt>
{"\n".join(response_blocks)}
Merge the responses into a coherent one. List any major conflicts.
'''


def build_ranking_question(batch_finals: Sequence[str]) -> str:
    return (
        "<context>\n"
        + "\n".join(batch_finals)
        + "\n</context>\nRank them into 4 tiers by moat width"
    )


async def query_panel(
    question: str,
    models: Models,
    clients: PanelClients,
    history: Sequence[HistoryItem] = (),
) -> PanelResponses:
    """Ask the full model panel concurrently."""
    gpt, gemini, claude, grok = await asyncio.gather(
        invoke_gpt(clients.gpt, models.gpt, question, history),
        invoke_gemini(clients.gemini, models.gemini, question, history),
        invoke_claude(clients.claude, models.claude, question, history),
        invoke_grok(clients.grok, models.grok, question, history),
    )
    return PanelResponses(gpt=gpt, gemini=gemini, claude=claude, grok=grok)


async def synthesize(
    question: str,
    responses: PanelResponses,
    models: Models,
    clients: PanelClients,
    history: Sequence[HistoryItem] = (),
) -> str:
    """Have GPT merge the panel's responses."""
    prompt = build_synthesis_prompt(question, responses)
    return await invoke_gpt(clients.gpt, models.gpt, prompt, history)


async def deliberate(
    question: str,
    models: Models,
    clients: PanelClients,
    history: Sequence[HistoryItem] = (),
) -> str:
    """Ask the panel, then return GPT's synthesis."""
    responses = await query_panel(question, models, clients, history)
    return await synthesize(question, responses, models, clients, history)


async def process_batch(
    batch: Sequence[str], batch_number: int, models: Models, clients: PanelClients
) -> str:
    initial_question = build_moat_question(batch)
    progress(f"Batch {batch_number}: first-pass moat analysis")
    first_synthesis = await deliberate(initial_question, models, clients)

    # In pareja.py, the next turn's history contains the prior user question and
    # GPT's displayed synthesis. Give every model that same shared history.
    # The requested next user turn is deliberately just these two words.
    deeper_history = ((initial_question, first_synthesis),)
    progress(f"Batch {batch_number}: deeper analysis")
    return await deliberate("think deeper", models, clients, deeper_history)


def read_records() -> List[str]:
    records = []
    for line_number, raw_line in enumerate(sys.stdin, start=1):
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
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Records per batch (default: 4)."
    )
    parser.add_argument("--gpt-model", default=MODEL_GPT_BASE)
    parser.add_argument("--gemini-model", default=MODEL_GEMINI)
    parser.add_argument("--grok-model", default=MODEL_GROK_BASE)
    parser.add_argument("--claude-model", default=MODEL_CLAUDE)
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

    return ApiKeys(gpt=gpt, gemini=gemini, grok=grok, claude=claude)


async def run(args: argparse.Namespace) -> str:
    if args.batch_size != 4:
        raise ValueError("batch size must be 4 for this workflow")

    records = read_records()
    if not records:
        raise ValueError("No non-empty CSV records were supplied on standard input.")

    models = Models(
        gpt=args.gpt_model,
        gemini=args.gemini_model,
        grok=args.grok_model,
        claude=args.claude_model,
    )
    api_keys = load_api_keys()
    batches = [
        records[index : index + args.batch_size]
        for index in range(0, len(records), args.batch_size)
    ]
    batch_semaphore = asyncio.Semaphore(4)

    async with (
        AsyncOpenAI(api_key=api_keys.gpt) as gpt_client,
        anthropic.AsyncAnthropic(api_key=api_keys.claude) as claude_client,
        AsyncClient(api_key=api_keys.grok) as grok_client,
    ):
        clients = PanelClients(
            gpt=gpt_client,
            gemini=genai.Client(api_key=api_keys.gemini),
            grok=grok_client,
            claude=claude_client,
        )

        async def process_batch_with_limit(
            batch_number: int, batch: Sequence[str]
        ) -> str:
            async with batch_semaphore:
                return await process_batch(batch, batch_number, models, clients)

        batch_finals = await asyncio.gather(
            *(
                process_batch_with_limit(batch_number, batch)
                for batch_number, batch in enumerate(batches, start=1)
            )
        )

        ranking_question = build_ranking_question(batch_finals)
        progress("Final ranking: sending all batch findings to the model panel")
        return await deliberate(ranking_question, models, clients)


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
