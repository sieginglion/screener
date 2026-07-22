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
    grok: str
    claude: str


@dataclass(frozen=True)
class PanelResponses:
    gpt: str
    gemini: str
    claude: str
    grok: str


def progress(message: str) -> None:
    """Keep stdout reserved for the final synthesis."""
    print(message, file=sys.stderr, flush=True)


async def invoke_gpt(model: str, question: str, history: Sequence[HistoryItem]) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ModelInvocationError("OPENAI_API_KEY is required.")

    messages: List[Dict[str, str]] = []
    for prior_question, prior_answer in history:
        messages.append({"role": "user", "content": prior_question})
        messages.append({"role": "assistant", "content": prior_answer})
    messages.append({"role": "user", "content": question})

    try:
        async with AsyncOpenAI(api_key=api_key) as client:
            response = await client.responses.create(
                model=model,
                input=messages,
                reasoning={"effort": "xhigh"},
            )
        response_error = getattr(response, "error", None)
        if response_error is not None:
            code = getattr(response_error, "code", "") or ""
            message = getattr(response_error, "message", "") or "Unknown model failure."
            code_text = f" [{code}]" if code else ""
            raise ModelInvocationError(f"GPT model response{code_text}: {message}")

        status = getattr(response, "status", None)
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = getattr(details, "reason", None) or "unknown reason"
            raise ModelInvocationError(f"GPT incomplete response ({reason}).")
        if status and status != "completed":
            raise ModelInvocationError(f"GPT response status {status!r}.")

        output_text = getattr(response, "output_text", "") or ""
        if not output_text.strip():
            raise ModelInvocationError("GPT response contained no text.")

        return output_text
    except ModelInvocationError:
        raise
    except Exception as exc:
        raise ModelInvocationError(f"Error invoking OpenAI API: {exc}") from exc


async def invoke_gemini(
    model: str, question: str, history: Sequence[HistoryItem]
) -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ModelInvocationError("GEMINI_API_KEY/GOOGLE_API_KEY is required.")

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
        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_content(
            model=model, contents=contents, config=config
        )

        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason:
            reason = getattr(block_reason, "value", block_reason)
            raise ModelInvocationError(f"Gemini prompt was blocked ({reason}).")

        candidates = getattr(response, "candidates", None)
        if not candidates:
            raise ModelInvocationError("Gemini response contained no candidates.")

        candidate = candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        finish_reason_value = getattr(finish_reason, "value", finish_reason)
        if finish_reason_value != "STOP":
            reason = finish_reason_value or "unknown reason"
            detail = getattr(candidate, "finish_message", None)
            detail_text = f": {detail}" if detail else ""
            raise ModelInvocationError(
                f"Gemini response stopped with {reason!r}{detail_text}"
            )

        output_text = getattr(response, "text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise ModelInvocationError("Gemini response contained no text.")
        return output_text
    except ModelInvocationError:
        raise
    except Exception as exc:
        raise ModelInvocationError(f"Error invoking Gemini API: {exc}") from exc


async def invoke_grok(model: str, question: str, history: Sequence[HistoryItem]) -> str:
    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        raise ModelInvocationError("XAI_API_KEY is required.")

    try:
        async with AsyncClient(api_key=api_key) as client:
            chat = client.chat.create(model=model, tools=[], reasoning_effort="high")
            for prior_question, prior_answer in history:
                chat.append(user(prior_question))
                chat.append(assistant(prior_answer))
            chat.append(user(question))
            response = await chat.sample()

        response_error = getattr(response, "error", None)
        if response_error is not None:
            code = getattr(response_error, "code", "") or ""
            message = getattr(response_error, "message", "") or "Unknown model failure."
            code_text = f" [{code}]" if code else ""
            raise ModelInvocationError(f"Grok model response{code_text}: {message}")

        finish_reason = getattr(response, "finish_reason", None)
        if finish_reason not in {"REASON_STOP", "STOP", "stop"}:
            raise ModelInvocationError(
                f"Grok response stopped with {finish_reason or 'unknown reason'!r}."
            )

        content = getattr(response, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ModelInvocationError("Grok response contained no text.")
        return content.strip()
    except ModelInvocationError:
        raise
    except Exception as exc:
        raise ModelInvocationError(f"Error invoking xAI API: {exc}") from exc


async def invoke_claude(
    model: str, question: str, history: Sequence[HistoryItem]
) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ModelInvocationError("ANTHROPIC_API_KEY is required.")

    messages = []
    for prior_question, prior_answer in history:
        messages.append({"role": "user", "content": prior_question})
        messages.append({"role": "assistant", "content": prior_answer})
    messages.append({"role": "user", "content": question})

    try:
        async with anthropic.AsyncAnthropic(api_key=api_key) as client:
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
    question: str, models: Models, history: Sequence[HistoryItem] = ()
) -> PanelResponses:
    """Ask the full model panel concurrently."""
    responses = await asyncio.gather(
        invoke_gpt(models.gpt, question, history),
        invoke_gemini(models.gemini, question, history),
        invoke_claude(models.claude, question, history),
        invoke_grok(models.grok, question, history),
    )
    return PanelResponses(*responses)


async def synthesize(
    question: str,
    responses: PanelResponses,
    models: Models,
    history: Sequence[HistoryItem] = (),
) -> str:
    """Have GPT merge the panel's responses."""
    prompt = build_synthesis_prompt(question, responses)
    return await invoke_gpt(models.gpt, prompt, history)


async def deliberate(
    question: str, models: Models, history: Sequence[HistoryItem] = ()
) -> str:
    """Ask the panel, then return GPT's synthesis."""
    responses = await query_panel(question, models, history)
    return await synthesize(question, responses, models, history)


async def process_batch(batch: Sequence[str], batch_number: int, models: Models) -> str:
    initial_question = build_moat_question(batch)
    progress(f"Batch {batch_number}: first-pass moat analysis")
    first_synthesis = await deliberate(initial_question, models)

    # In pareja.py, the next turn's history contains the prior user question and
    # GPT's displayed synthesis. Give every model that same shared history.
    # The requested next user turn is deliberately just these two words.
    deeper_history = ((initial_question, first_synthesis),)
    progress(f"Batch {batch_number}: deeper analysis")
    return await deliberate("think deeper", models, deeper_history)


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
    batches = [
        records[index : index + args.batch_size]
        for index in range(0, len(records), args.batch_size)
    ]
    batch_semaphore = asyncio.Semaphore(4)

    async def process_batch_with_limit(batch_number: int, batch: Sequence[str]) -> str:
        async with batch_semaphore:
            return await process_batch(batch, batch_number, models)

    batch_finals = await asyncio.gather(
        *(
            process_batch_with_limit(batch_number, batch)
            for batch_number, batch in enumerate(batches, start=1)
        )
    )

    ranking_question = build_ranking_question(batch_finals)
    progress("Final ranking: sending all batch findings to the model panel")
    return await deliberate(ranking_question, models)


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
