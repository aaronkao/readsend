"""
Script to add 'source: Kindle' metadata to existing Kindle highlights in Pinecone.
This is a one-time migration for vectors that were ingested before the source field was added.
"""
import os
import sys
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

def migrate_kindle_source():
    validate_env()
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_INDEX_HOST)
    
    print("Fetching all vectors without 'source' metadata...")
    
    # List all vectors (paginated)
    # Pinecone serverless supports list() to get all vector IDs
    updated_count = 0
    batch_size = 100
    
    # Get all vector IDs
    all_ids = []
    for ids_batch in index.list():
        all_ids.extend(ids_batch)
    
    print(f"Found {len(all_ids)} total vectors in index.")
    
    # Process in batches to fetch and update
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        
        # Fetch vectors with metadata
        fetch_result = index.fetch(ids=batch_ids)
        
        updates = []
        for vec_id, vec_data in fetch_result.vectors.items():
            metadata = vec_data.metadata or {}
            
            # Skip if already has source (Bible verses)
            if 'source' in metadata:
                continue
            
            # This is a Kindle highlight - add source
            metadata['source'] = 'Kindle'
            updates.append({
                "id": vec_id,
                "values": vec_data.values,
                "metadata": metadata
            })
        
        if updates:
            index.upsert(vectors=updates)
            updated_count += len(updates)
            print(f"Updated {len(updates)} vectors in batch {i//batch_size + 1}")
    
    print(f"\nMigration complete! Updated {updated_count} Kindle highlights with source metadata.")

if __name__ == "__main__":
    migrate_kindle_source()
