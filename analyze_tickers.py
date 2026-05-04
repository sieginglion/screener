#!/usr/bin/env python3
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List

from dotenv import load_dotenv

# Load env before importing other libs to ensure keys are available
load_dotenv()

from openai import OpenAI
from xai_sdk import Client as XAIClient
from xai_sdk.chat import system, user
from xai_sdk.tools import web_search, x_search

from config import MARKET

# --- Configuration ---
MODEL_GPT = "gpt-5.5"
MODEL_GROK = "grok-4.3"
BATCH_SIZE = 8
CONCURRENCY = 8
SYSTEM_INSTRUCTION = "You are an equity research analyst."
LANGUAGE_SUFFIX_BY_MARKET = {
    "t": ". In Taiwanese Chinese",
    "j": ". In Taiwanese Chinese",
}


def check_env():
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("XAI_API_KEY"):
        missing.append("XAI_API_KEY")

    if missing:
        sys.stderr.write(
            f"Error: Missing environment variables: {', '.join(missing)}\n"
        )
        sys.exit(1)


def read_input() -> List[str]:
    if len(sys.argv) != 1:
        sys.stderr.write("Usage: python analyze_tickers.py < stdin\n")
        sys.exit(1)

    def die(message: str) -> None:
        sys.stderr.write(f"Error: {message}\n")
        sys.exit(1)

    tickers = []
    for line_number, row in enumerate(csv.reader(sys.stdin), start=1):
        if len(row) < 2:
            die(f"line {line_number} must include at least: symbol,description")

        symbol = row[0].strip()
        description = row[1].strip()

        if symbol == "symbol" and description == "description":
            continue

        if not symbol or not description:
            die(f"line {line_number} has an empty field")

        tickers.append(f"{symbol} {description}")

    if not tickers:
        sys.exit(0)

    return tickers


def batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def invoke_grok(client: XAIClient, prompt: str) -> str:
    """Invoke Grok reasoning with web search and x_search enabled."""
    try:
        tools = [web_search(), x_search()]

        # Create chat with tools
        chat = client.chat.create(model=MODEL_GROK, tools=tools)

        chat.append(system(SYSTEM_INSTRUCTION))
        chat.append(user(prompt))

        # Reasoning models reason automatically; do not pass reasoning_effort.
        response = chat.sample()
        return response.content
    except Exception as e:
        sys.stderr.write(f"Error invoking Grok: {e}\n")
        raise


def invoke_gpt(client: OpenAI, prompt: str, reasoning_effort: str) -> str:
    """Invoke GPT-5.2 with specific reasoning effort and NO tools."""
    try:
        # Using client.responses.create as per mini_gpt.py (adapted for sync)
        response = client.responses.create(
            model=MODEL_GPT,
            input=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            reasoning={"effort": reasoning_effort},
            # tools parameter explicitly OMITTED
        )

        # Extract text based on mini_gpt.py pattern
        text = getattr(response, 'output_text', str(response))
        return text
    except Exception as e:
        sys.stderr.write(f"Error invoking GPT: {e}\n")
        raise


def analyze_batch(
    tickers_batch: List[str], grok_client: XAIClient, language_suffix: str
) -> str:
    # --- Step A: Description (Web Search Enabled via Grok) ---
    tickers_multiline = "\n".join(tickers_batch)

    grok_prompt = f"<text>\n{tickers_multiline}\n</text>\nSearch then tell me what they do. Detailed yet concise. Include ticker symbol and company name{language_suffix}"

    try:
        descriptions = invoke_grok(grok_client, grok_prompt)
        sys.stderr.write(f"[Step A Output]\n{descriptions}\n[/Step A Output]\n")
        return descriptions
    except Exception:
        return ""


def main():
    check_env()
    tickers = read_input()

    # Initialize Clients
    try:
        grok_client = XAIClient(api_key=os.getenv("XAI_API_KEY"), timeout=3600)
        gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        sys.stderr.write(f"Error initializing clients: {e}\n")
        sys.exit(1)

    batches = list(batched(tickers, BATCH_SIZE))
    language_suffix = LANGUAGE_SUFFIX_BY_MARKET.get(MARKET, "")

    # Process batches concurrently
    all_batch_results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        # We need to pass clients to the worker.
        # OpenAI and XAI clients should be thread-safe for making requests,
        # or we instantiate them inside?
        # Usually standard clients are thread-safe.
        futures = [
            executor.submit(analyze_batch, batch, grok_client, language_suffix)
            for batch in batches
        ]

        for future in futures:
            try:
                result = future.result()
                if result:
                    all_batch_results.append(result)
            except Exception as e:
                sys.stderr.write(f"Batch execution error: {e}\n")

    if all_batch_results:
        # --- Reduce / Merge Phase ---
        all_results_str = "\n\n".join(all_batch_results)

        reduce_prompt = f"<text>\n{all_results_str}\n</text>\nGroup them by what they do. One can be in multiple groups. Ensure no one is overlooked. Include ticker symbol and company name{language_suffix}"

        try:
            # Step C: GPT, High Reasoning
            final_report = invoke_gpt(
                gpt_client, reduce_prompt, reasoning_effort="high"
            )
            print(final_report)
        except Exception as e:
            sys.stderr.write(f"Error in Reduce Phase: {e}\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
