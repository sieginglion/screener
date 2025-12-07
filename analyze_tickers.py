import sys
import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration
MODEL_NAME = "gpt-5.1"
BATCH_SIZE = 8
TEMPERATURE = 0.8
SERVICE_TIER = "priority"

def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.stderr.write("Error: OPENAI_API_KEY environment variable not found.\n")
        sys.exit(1)
    
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        service_tier=SERVICE_TIER,
        api_key=api_key
    )

def chunk_tickers(tickers: List[str], size: int):
    """Yield successive n-sized chunks from tickers."""
    for i in range(0, len(tickers), size):
        yield tickers[i:i + size]

def process_batch(llm, tickers_batch: List[str], system_instruction: str) -> str:
    # Step A: Description
    step_a_prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "<text>\n{tickers_multiline}\n</text>\nWhat do they do?")
    ])
    
    tickers_multiline = "\n".join(tickers_batch)
    chain_a = step_a_prompt | llm | StrOutputParser()
    descriptions = chain_a.invoke({"tickers_multiline": tickers_multiline})
    sys.stderr.write(f"\n[DEBUG] Batch Step A (Descriptions):\n{descriptions}\n")

    # Step B: Categorization
    step_b_prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "<text>\n{descriptions}\n</text>\nCategorize them by industry")
    ])
    
    chain_b = step_b_prompt | llm | StrOutputParser()
    categorization = chain_b.invoke({"descriptions": descriptions})
    sys.stderr.write(f"\n[DEBUG] Batch Step B (Categorization):\n{categorization}\n")
    
    return categorization

def reduce_results(llm, all_results: List[str], system_instruction: str) -> str:
    reduce_prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "<text>\n{all_results}\n</text>\nCombine them")
    ])
    
    combined_text = "\n\n".join(all_results)
    chain_reduce = reduce_prompt | llm | StrOutputParser()
    final_output = chain_reduce.invoke({"all_results": combined_text})
    
    return final_output

def main():
    # Load environment variables
    load_dotenv()

    # Input handling: Check for TTY
    if sys.stdin.isatty():
        sys.stderr.write("Usage: echo 'TICKER1;TICKER2' | python analyze_tickers.py\n")
        sys.exit(1)

    # Read all input from stdin
    try:
        input_str = sys.stdin.read()
    except Exception as e:
        sys.stderr.write(f"Error reading input: {e}\n")
        sys.exit(1)

    # Normalize input
    # Spec: Split by ';' (no trim/filter yet, but usually stripping the final newline of the input itself is safe, 
    # but the spec says "Split rule: tickers = input_str.split(';') (no trim/filter)".
    # However, if input ends with newline, the last element might be distinct.
    # The spec is strict: "no trim/filter". I will adhere to that strictly for the splitting logic itself.
    # But usually `input_str` might have a trailing newline from cat/echo. 
    # I'll rely on the split as requested.
    tickers = input_str.split(";")
    
    # Filter out empty strings if they are just artifacts of trailing semi-colons or newlines if strictly desired?
    # Spec says "no trim/filter". So I will pass them as is. 
    # Wait, if I pass empty string to LLM it might be weird. 
    # But "no trim/filter" implies I should respect the spec. 
    # However, standard practice often involves removing empty inputs.
    # I'll assume standard defensive coding for EMPTY tickers is acceptable if they are effectively whitespace.
    # Actually, let's stick to the spec literalness for the split, but maybe the Batch logic handles it?
    # If a ticker is "\n", passing it to LLM might be fine.
    
    llm = get_llm()
    system_instruction = "You are a buy-side analyst."
    
    # Batch loops
    all_step_b_results = []
    
    # Create chunks
    # Note: If the split resulted in empty strings, we still process them as per spec "no filter".
    # Iterate through batches
    for batch in chunk_tickers(tickers, BATCH_SIZE):
        if not batch:
            continue
            
        try:
            result = process_batch(llm, batch, system_instruction)
            all_step_b_results.append(result)
        except Exception as e:
            sys.stderr.write(f"Error processing batch {batch}: {e}\n")
            sys.exit(1)

    if not all_step_b_results:
        # If input was empty or resulted in no processing
        return

    # Reduce Phase
    try:
        final_report = reduce_results(llm, all_step_b_results, system_instruction)
        print(final_report)
    except Exception as e:
        sys.stderr.write(f"Error in reduce phase: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
