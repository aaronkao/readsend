import os
import sys
import random
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

def validate_env():
    if not all([PINECONE_API_KEY, PINECONE_INDEX_HOST]):
        print("Error: Missing environment variables.")
        print("Please set PINECONE_API_KEY and PINECONE_INDEX_HOST in .env")
        sys.exit(1)

def get_random_vector(dim=1536):
    """Generate a random unit vector."""
    return [random.uniform(-1, 1) for _ in range(dim)]

def get_random_highlights(index, count=5):
    """
    Fetch 'count' random highlights.
    Performs independent queries to ensure variety, avoiding clusters.
    """
    highlights = []
    seen_ids = set()
    
    # Try to get 'count' unique highlights
    # We loop up to count * 2 times to avoid infinite loops if index is small
    attempts = 0
    while len(highlights) < count and attempts < count * 2:
        attempts += 1
        query_vector = get_random_vector(1024)
        try:
            results = index.query(
                vector=query_vector,
                top_k=1,
                include_metadata=True
            )
            
            if results['matches']:
                match = results['matches'][0]
                if match['id'] not in seen_ids:
                    highlights.append(match)
                    seen_ids.add(match['id'])
                    
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            continue
            
    return highlights

def main():
    validate_env()
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_INDEX_HOST)
    
    print("Fetching your daily random highlights...")
    
    highlights = get_random_highlights(index, count=5)
    
    print(f"\nFound {len(highlights)} highlights:\n")
    
    for i, match in enumerate(highlights, 1):
        metadata = match['metadata']
        print(f"{i}. {metadata.get('title', 'Unknown Title')} ({metadata.get('author', 'Unknown Author')})")
        print(f"   \"{metadata.get('content', '').strip()}\"")
        print("-" * 40)

if __name__ == "__main__":
    main()
