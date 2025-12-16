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

def get_random_highlights(index, count=5, source=None):
    """
    Fetch 'count' random highlights.
    Performs independent queries to ensure variety, avoiding clusters.
    
    Args:
        index: Pinecone index
        count: Number of highlights to fetch
        source: Optional filter - 'Kindle' or 'Bible'
    """
    highlights = []
    seen_ids = set()
    
    # Build filter if source specified
    filter_dict = {"source": {"$eq": source}} if source else None
    
    # Try to get 'count' unique highlights
    # We loop up to count * 3 times to avoid infinite loops
    attempts = 0
    while len(highlights) < count and attempts < count * 3:
        attempts += 1
        query_vector = get_random_vector(1024)
        try:
            results = index.query(
                vector=query_vector,
                top_k=1,
                include_metadata=True,
                filter=filter_dict
            )
            
            if results['matches']:
                match = results['matches'][0]
                if match['id'] not in seen_ids:
                    content = match['metadata'].get('content', '').strip()
                    # Filter out one-word highlights
                    if len(content.split()) > 1:
                        highlights.append(match)
                        seen_ids.add(match['id'])
                    
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            continue
            
    return highlights

def get_kindle_highlights(index, count=5):
    """Fetch random Kindle highlights."""
    return get_random_highlights(index, count=count, source="Kindle")

def get_bible_verses(index, count=2):
    """Fetch random Bible verses."""
    return get_random_highlights(index, count=count, source="Bible")

def main():
    validate_env()
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_INDEX_HOST)
    
    print("Fetching your daily random highlights...")
    
    kindle_highlights = get_kindle_highlights(index, count=5)
    bible_verses = get_bible_verses(index, count=2)
    
    all_highlights = kindle_highlights + bible_verses
    
    print(f"\nFound {len(kindle_highlights)} Kindle highlights and {len(bible_verses)} Bible verses:\n")
    
    for i, match in enumerate(all_highlights, 1):
        metadata = match['metadata']
        source_tag = f"[{metadata.get('source', 'Unknown')}]"
        print(f"{i}. {source_tag} {metadata.get('title', 'Unknown Title')} ({metadata.get('author', 'Unknown Author')})")
        print(f"   \"{metadata.get('content', '').strip()}\"")
        print("-" * 40)

if __name__ == "__main__":
    main()

