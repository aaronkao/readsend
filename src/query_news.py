import os
import sys
import argparse
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "aaron")

def validate_env():
    missing = []
    if not PINECONE_API_KEY: missing.append("PINECONE_API_KEY")
    if not PINECONE_INDEX_HOST: missing.append("PINECONE_INDEX_HOST")
    
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("Please set them in .env")
        sys.exit(1)

def get_embedding(text: str, pc: Pinecone) -> List[float]:
    """Generate embedding for a single text using Pinecone Inference API."""
    response = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[text],
        parameters={"input_type": "query", "truncate": "END"}
    )
    return response[0]['values']

def query_library(summary: str, top_k: int = 5):
    validate_env()
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_INDEX_HOST)
    
    print(f"Generating embedding for summary...")
    query_vec = get_embedding(summary, pc)
    
    print(f"Querying library in namespace '{PINECONE_NAMESPACE}'...")
    results = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE
    )
    
    if not results['matches']:
        print("No matches found.")
        return
        
    print(f"\nTop {len(results['matches'])} matches for: \"{summary[:100]}...\"\n")
    
    for i, match in enumerate(results['matches'], 1):
        metadata = match['metadata']
        score = match['score']
        source = metadata.get('source', 'Unknown')
        title = metadata.get('title', 'Unknown Title')
        author = metadata.get('author', 'Unknown Author')
        content = metadata.get('content', '').strip()
        
        print(f"{i}. [{source}] {title} ({author}) - Score: {score:.4f}")
        print(f"   \"{content}\"")
        print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Query your library with a news summary.")
    parser.add_argument("summary", type=str, nargs='?', help="The news summary to query with.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return.")
    
    args = parser.parse_args()
    
    summary = args.summary
    if not summary:
        print("Please enter a news summary (or paste text):")
        summary = sys.stdin.read().strip()
        
    if not summary:
        print("Error: No summary provided.")
        sys.exit(1)
        
    query_library(summary, top_k=args.top_k)

if __name__ == "__main__":
    main()
