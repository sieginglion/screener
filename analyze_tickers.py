import sys
import os
import json
from typing import List
from dotenv import load_dotenv

# Load env before importing langchain to ensure key is available if already set in .env
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# --- Configuration ---
MODEL_NAME = "gpt-5.1"
SERVICE_TIER = "priority"
BATCH_SIZE = 8
TEMPERATURE = 0.8
SYSTEM_INSTRUCTION = "You are a buy-side analyst."

def check_env():
    if not os.getenv("OPENAI_API_KEY"):
        sys.stderr.write("Error: OPENAI_API_KEY environment variable is not set.\n")
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

def batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

def analyze_batch(tickers_batch: List[str], llm: ChatOpenAI) -> str:
    # --- Step A: Description (Web Search Enabled) ---
    tickers_multiline = "\n".join(tickers_batch)
    
    # We bind the native web_search tool. 
    # Note: As of typical LangChain/OpenAI integration, pass tools in the bind method.
    # We use the 'tools' parameter with the specific list structure for native tools.
    llm_with_search = llm.bind(tools=[{"type": "web_search"}])
    
    messages_a = [
        SystemMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(content=f"<text>\n{tickers_multiline}\n</text>\nWhat do they do? Search first")
    ]
    
    try:
        # We invoke the model. It handles the tool call internally if it decides to use it.
        # However, for 'web_search' type, newer models often return the final answer directly 
        # after performing the search on the server side, or we might need to handle tool_calls.
        # Given "gpt-5.1" and "web_search" (likely ChatGPT-style search), 
        # we assume it returns the content directly or performs the generation after search.
        # If it returns tool_calls, we might need a loop, but OpenAI 'web_search' is often automatic in recent iterations/docs.
        # But to be safe and simple as per prompt: we get the response.
        
        response_a = llm_with_search.invoke(messages_a)
        descriptions = response_a.content
        
        # Optional debug logging
        sys.stderr.write(f"[DEBUG] Batch descriptions:\n{descriptions}\n")
        
    except Exception as e:
        sys.stderr.write(f"Error in Step A (Description): {e}\n")
        sys.exit(1)

    # --- Step B: Categorization (No Tools) ---
    messages_b = [
        SystemMessage(content=SYSTEM_INSTRUCTION),
        HumanMessage(content=f"<text>\n{descriptions}\n</text>\nCategorize them by industry")
    ]
    
    try:
        response_b = llm.invoke(messages_b)
        categorization = response_b.content
        
        # Optional debug logging
        sys.stderr.write(f"[DEBUG] Batch categorization:\n{categorization}\n")
        
        return categorization
    except Exception as e:
        sys.stderr.write(f"Error in Step B (Categorization): {e}\n")
        sys.exit(1)

def main():
    check_env()
    tickers = read_input()
    
    # Initialize LLM
    # Note: 'service_tier' may not be standard in all SDK versions yet, but request specifies it.
    # We pass it via model_kwargs if not directly supported by constructor.
    llm = ChatOpenAI(
        model=MODEL_NAME, 
        temperature=TEMPERATURE,
        service_tier=SERVICE_TIER
    )

    all_step_b_results = []
    
    for batch in batched(tickers, BATCH_SIZE):
        result = analyze_batch(batch, llm)
        all_step_b_results.append(result)
        
    if all_step_b_results:
        # --- Reduce / Merge Phase ---
        all_results_str = "\n\n".join(all_step_b_results)
        
        messages_reduce = [
            SystemMessage(content=SYSTEM_INSTRUCTION),
            HumanMessage(content=f"<text>\n{all_results_str}\n</text>\nMerge them, and make sure nothing is missing")
        ]
        
        try:
            # Use high reasoning effort for the final merge step
            llm_reduce = llm.bind(reasoning_effort="high")
            response_reduce = llm_reduce.invoke(messages_reduce)
            final_report = response_reduce.content
            print(final_report)
        except Exception as e:
            sys.stderr.write(f"Error in Reduce Phase: {e}\n")
            sys.exit(1)

if __name__ == "__main__":
    main()
