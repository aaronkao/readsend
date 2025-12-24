import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "aaron")

def validate_env():
    if not all([PINECONE_API_KEY, PINECONE_INDEX_HOST]):
        print("Error: Missing environment variables.")
        print("Please set PINECONE_API_KEY and PINECONE_INDEX_HOST in .env")
        sys.exit(1)

def migrate_to_namespace(target_namespace=PINECONE_NAMESPACE):
    validate_env()
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_INDEX_HOST)
    
    print(f"Migrating vectors from default namespace to '{target_namespace}'...")
    
    # List all vectors in the default namespace
    all_ids = []
    try:
        for ids_batch in index.list(namespace=""):
            all_ids.extend(ids_batch)
    except Exception as e:
        print(f"Error listing vectors: {e}")
        return

    if not all_ids:
        print("No vectors found in default namespace.")
        return

    print(f"Found {len(all_ids)} total vectors to migrate.")
    
    batch_size = 100
    migrated_count = 0
    
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        
        # Fetch vectors with metadata and values
        fetch_result = index.fetch(ids=batch_ids, namespace="")
        
        vectors_to_upsert = []
        for vec_id, vec_data in fetch_result.vectors.items():
            vectors_to_upsert.append({
                "id": vec_id,
                "values": vec_data.values,
                "metadata": vec_data.metadata
            })
        
        if vectors_to_upsert:
            # Upsert to target namespace
            index.upsert(vectors=vectors_to_upsert, namespace=target_namespace)
            
            # Delete from default namespace after successful upsert
            index.delete(ids=batch_ids, namespace="")
            
            migrated_count += len(vectors_to_upsert)
            print(f"Migrated {migrated_count}/{len(all_ids)} vectors...")
    
    print(f"\nMigration complete! Successfully moved {migrated_count} vectors to namespace '{target_namespace}'.")

if __name__ == "__main__":
    confirm = input(f"This will move all vectors from the default namespace to '{PINECONE_NAMESPACE}'. Continue? (y/n): ")
    if confirm.lower() == 'y':
        migrate_to_namespace()
    else:
        print("Migration cancelled.")
