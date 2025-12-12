import sys
import os
import json
from typing import List
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load env before importing langchain to ensure key is available if already set in .env
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# --- Configuration ---
MODEL_NAME = "openai/gpt-5.2"
MODEL_NAME_SEARCH = "x-ai/grok-4.1-fast"
BASE_URL = "https://openrouter.ai/api/v1"
BATCH_SIZE = 8
CONCURRENCY = 2
TEMPERATURE = 0.7
SYSTEM_INSTRUCTION = "You are a buy-side analyst."

def check_env():
    if not os.getenv("OPENROUTER_API_KEY"):
        sys.stderr.write("Error: OPENROUTER_API_KEY environment variable is not set.\n")
        sys.exit(1)

def read_input() -> List[str]:
    if sys.stdin.isatty():
        sys.stderr.write("Usage: echo 'TICKER1;TICKER2' | python analyze_tickers.py\n")
        sys.exit(1)
    
    try:
        input_str = sys.stdin.read()
    except Exception as e:
        sys.stderr.write(f"Error reading from stdin: {e}\n")
        sys.exit(1)

    raw_tickers = input_str.split(";")
    tickers = [t.strip() for t in raw_tickers]
    tickers = [t for t in tickers if t] # Drop empty strings
    
    if not tickers:
        sys.exit(0)
        
    return tickers

def invoke_llm(llm: ChatOpenAI, messages: List) -> str:
    """Helper to invoke LLM and ensure string output."""
    response = llm.invoke(messages)
    content = response.content
    
    # Both models return list of dicts - extract text from items that have it
    result = "\n".join([item["text"] for item in content if "text" in item])
    
    sys.stderr.write(f"[invoke_llm output]\n{result}\n[/invoke_llm output]\n")
    return result

def batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

def analyze_batch(tickers_batch: List[str], llm_search: ChatOpenAI, llm: ChatOpenAI) -> str:
    # --- Step A: Description (Web Search Enabled via Grok) ---
    tickers_multiline = "\n".join(tickers_batch)
    
    messages_a = [
        SystemMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(content=f"<text>\n{tickers_multiline}\n</text>\nSearch then tell me what they do")
    ]
    
    try:
        descriptions = invoke_llm(llm_search, messages_a)
    except Exception as e:
        sys.stderr.write(f"Error in Step A (Description): {e}\n")
        sys.exit(1)

    # --- Step B: Categorization (No Tools) ---
    messages_b = [
        SystemMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(content=f"<text>\n{descriptions}\n</text>\nCategorize them by industry")
    ]
    
    try:
        categorization = invoke_llm(llm, messages_b)
        return categorization
    except Exception as e:
        sys.stderr.write(f"Error in Step B (Categorization): {e}\n")
        sys.exit(1)

def main():
    check_env()
    tickers = read_input()
    
    # Shared configuration
    common_params = {
        "temperature": TEMPERATURE,
        "base_url": BASE_URL,
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "use_responses_api": True,
    }

    # Initialize LLMs via OpenRouter
    llm_search = ChatOpenAI(
        model=MODEL_NAME_SEARCH,
        **common_params,
        model_kwargs={"tools": [{"type": "web_search"}]},
    )
    llm = ChatOpenAI(
        model=MODEL_NAME,
        **common_params,
    )

    batches = list(batched(tickers, BATCH_SIZE))
    
    # Process batches concurrently
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        all_step_b_results = list(executor.map(
            lambda batch: analyze_batch(batch, llm_search, llm),
            batches
        ))
        
    if all_step_b_results:
        # --- Reduce / Merge Phase ---
        all_results_str = "\n\n".join(all_step_b_results)
        
        messages_reduce = [
            SystemMessage(content=SYSTEM_INSTRUCTION),
            HumanMessage(content=f"<text>\n{all_results_str}\n</text>\nMerge them and ensure nothing is missing")
        ]
        
        try:
            # Use high reasoning effort for the final merge step
            final_report = invoke_llm(llm.bind(reasoning={"effort": "high"}), messages_reduce)
            print(final_report)
        except Exception as e:
            sys.stderr.write(f"Error in Reduce Phase: {e}\n")
            sys.exit(1)

if __name__ == "__main__":
    main()
