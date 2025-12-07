import os
import sys
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def chunk_list(lst: List[str], chunk_size: int) -> List[List[str]]:
    """Split a list into chunks of a given size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def analyze_tickers():
    load_dotenv()
    
    # check api key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    # Get input
    print("Enter a list of ticker symbols (comma-separated):")
    user_input = input().strip()
    if not user_input:
        print("No input provided.")
        return

    tickers = [t.strip() for t in user_input.split(",") if t.strip()]
    print(f"\nProcessing {len(tickers)} tickers...")

    # Initialize model
    model = ChatOpenAI(model="gpt-4o")
    parser = StrOutputParser()

    # Define prompts
    description_template = """What do the following companies do?
Tickers: {tickers}

Please provide a brief description for each."""
    
    categorization_template = """Based on the following descriptions, categorize these companies by industry.
Descriptions:
{descriptions}

Output format:
- Ticker: Industry (Brief Note)"""

    combine_template = """Combine the following industry categorization lists into a single consolidated summary. Group them by industry.

Lists:
{categorizations}
"""

    description_prompt = PromptTemplate.from_template(description_template)
    categorization_prompt = PromptTemplate.from_template(categorization_template)
    combine_prompt = PromptTemplate.from_template(combine_template)

    # Chains
    desc_chain = description_prompt | model | parser
    cat_chain = categorization_prompt | model | parser
    combine_chain = combine_prompt | model | parser

    # Process in chunks
    chunk_size = 8
    ticker_chunks = chunk_list(tickers, chunk_size)
    
    all_categorizations = []

    print("\n--- Starting Batch Analysis ---")

    for i, chunk in enumerate(ticker_chunks):
        chunk_str = ", ".join(chunk)
        print(f"\nBatch {i+1}/{len(ticker_chunks)}: {chunk_str}")
        
        # Step 1: Describe
        print("  > Fetching descriptions...")
        descriptions = desc_chain.invoke({"tickers": chunk_str})
        
        # Step 2: Categorize
        print("  > Categorizing...")
        categorization = cat_chain.invoke({"descriptions": descriptions})
        
        print("\n  [Batch Attempt Result]")
        print(categorization)
        
        all_categorizations.append(categorization)

    # Final Step: Combine
    if all_categorizations:
        print("\n--- Final Combined Analysis ---")
        print("Combining results...")
        combined_result = combine_chain.invoke({"categorizations": "\n\n".join(all_categorizations)})
        print("\n" + combined_result)

if __name__ == "__main__":
    analyze_tickers()
