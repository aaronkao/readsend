import os
import sys
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from parser import ClippingsParser, Highlight

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

def validate_env():
    if not all([PINECONE_API_KEY, PINECONE_INDEX_HOST]):
        print("Error: Missing environment variables.")
        print("Please set PINECONE_API_KEY and PINECONE_INDEX_HOST in .env")
        sys.exit(1)

def get_embeddings_batch(texts: List[str], pc: Pinecone) -> List[List[float]]:
    """Generate embeddings using Pinecone Inference API."""
    # Model: multilingual-e5-large (1024 dimensions)
    response = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=texts,
        parameters={"input_type": "passage", "truncate": "END"}
    )
    return [r['values'] for r in response]

def batch_upsert(index, vectors, batch_size=100):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i//batch_size + 1}")

def main():
    validate_env()
    
    # Initialize clients
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_INDEX_HOST)
    
    # Parse clippings
    parser = ClippingsParser()
    print("Parsing My Clippings.txt...")
    try:
        highlights = parser.parse("My Clippings.txt")
    except FileNotFoundError:
        print("Error: 'My Clippings.txt' not found.")
        sys.exit(1)
        
    print(f"Found {len(highlights)} highlights.")
    
    vectors = []
    print("Generating embeddings via Pinecone Inference...")
    
    # Process in batches to respect API limits and efficiency
    batch_size = 10  # Inference API batch size
    
    for i in range(0, len(highlights), batch_size):
        batch_highlights = highlights[i:i + batch_size]
        batch_texts = [h.content for h in batch_highlights]
        
        try:
            embeddings = get_embeddings_batch(batch_texts, pc)
            
            for h, emb in zip(batch_highlights, embeddings):
                metadata = {
                    "title": h.title,
                    "author": h.author,
                    "content": h.content,
                    "original_metadata": h.metadata
                }
                vectors.append({
                    "id": h.id,
                    "values": emb,
                    "metadata": metadata
                })
            
            print(f"Processed {min(i + batch_size, len(highlights))}/{len(highlights)}")
            
        except Exception as e:
            print(f"Error processing batch starting at {i}: {e}")

    if vectors:
        print(f"Upserting {len(vectors)} vectors to Pinecone...")
        batch_upsert(index, vectors)
        print("Ingestion complete!")
    else:
        print("No vectors to upsert.")

if __name__ == "__main__":
    main()
