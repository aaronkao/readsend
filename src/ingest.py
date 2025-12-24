import os
import sys
import json
import time
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone
from parser import ClippingsParser
from esv_sdk import Esv

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "aaron")
ESV_API_KEY = os.getenv("ESV_API_KEY")

def validate_env():
    missing = []
    if not PINECONE_API_KEY: missing.append("PINECONE_API_KEY")
    if not PINECONE_INDEX_HOST: missing.append("PINECONE_INDEX_HOST")
    
    if missing:
        print(f"Error: Missing environment variables: {', '.join(missing)}")
        print("Please set them in .env")
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
        index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
        print(f"Upserted batch {i//batch_size + 1}")

def ingest_clippings(index, pc):
    """Ingest highlights from 'My Clippings.txt'."""
    print("\n--- Processing My Clippings.txt ---")
    parser = ClippingsParser()
    try:
        highlights = parser.parse("My Clippings.txt")
    except FileNotFoundError:
        print("Error: 'My Clippings.txt' not found. Skipping.")
        return
        
    print(f"Found {len(highlights)} highlights.")
    
    vectors = []
    batch_size = 10  # Inference API batch size
    
    for i in range(0, len(highlights), batch_size):
        batch_highlights = highlights[i:i + batch_size]
        batch_texts = [h.content for h in batch_highlights]
        
        try:
            embeddings = get_embeddings_batch(batch_texts, pc)
            
            for h, emb in zip(batch_highlights, embeddings):
                metadata = {
                    "source": "Kindle",
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
        print(f"Upserting {len(vectors)} clippings to Pinecone...")
        batch_upsert(index, vectors)
    else:
        print("No clippings vectors to upsert.")

def fetch_bible_verses():
    """Ingest Bible verses from 'highlight_references.json'."""
    print("\n--- Processing Bible Verses ---")
    
    if not ESV_API_KEY:
        print("Warning: ESV_API_KEY not set. Skipping Bible verse ingestion.")
        return

    try:
        with open("highlight_references.json", "r") as f:
            references = json.load(f)
    except FileNotFoundError:
        print("Error: 'highlight_references.json' not found. Skipping.")
        return

    print(f"Found {len(references)} references.")
    
    vectors = []
    # Process individually to be safe with ESV API rate limits if necessary, 
    # but we can batch the embedding part.
    # For simplicity and robustness, we'll fetch text one by one, then batch embed.
    
    # Check if we should process a subset or all. 
    # We will process all but print progress.
    
    texts_to_embed = []
    metadatas_to_embed = []
    ids_to_embed = []
    
    # Load already fetched verses to avoid re-fetching
    output_file = "fetched_verses.json"
    fetched_verses = {}
    try:
        with open(output_file, "r") as f:
            fetched_verses = {v['reference']: v for v in json.load(f)}
        print(f"Loaded {len(fetched_verses)} already fetched verses from {output_file}")
    except FileNotFoundError:
        pass

    # Initialize ESV client
    with Esv(api_key=ESV_API_KEY) as esv:
        for i, ref_obj in enumerate(references):
            ref_str = ref_obj.get("reference_human")
            version_abbrev = ref_obj.get("version_abbreviation", "ESV")

            if not ref_str:
                continue
            
            # Skip if already fetched
            if ref_str in fetched_verses:
                data = fetched_verses[ref_str]
                texts_to_embed.append(data['content'])
                metadatas_to_embed.append(data)
                ids_to_embed.append(data['id'])
                continue

            # Retry logic with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Use ESV SDK to get text
                    response = esv.passages.get_text(
                        query=ref_str,
                        include_passage_references=False,
                        include_verse_numbers=False,
                        include_footnotes=False,
                        include_headings=False,
                        include_short_copyright=False
                    )
                    
                    if not response.passages or not response.passages[0]:
                        print(f"Warning: No text returned for {ref_str}")
                        break
                    
                    passage_text = response.passages[0].strip()

                    # Construct ID
                    safe_ref = ref_str.replace(" ", "_").replace(":", "_").replace("-", "_")
                    vec_id = f"bible_{version_abbrev}_{safe_ref}"

                    metadata = {
                        "id": vec_id,
                        "source": "Bible",
                        "title": ref_str,
                        "author": "Bible",
                        "content": passage_text,
                        "book": "Unknown",
                        "chapter": "Unknown",
                        "verse": "Unknown",
                        "reference": ref_str,
                        "version": "ESV"
                    }
                    
                    # Simple parsing attempt
                    parts = ref_str.rsplit(' ', 1)
                    if len(parts) == 2:
                        metadata['book'] = parts[0]
                        numerals = parts[1].split(':')
                        if len(numerals) >= 1: metadata['chapter'] = numerals[0]
                        if len(numerals) >= 2: metadata['verse'] = numerals[1]

                    texts_to_embed.append(passage_text)
                    metadatas_to_embed.append(metadata)
                    ids_to_embed.append(vec_id)
                    fetched_verses[ref_str] = metadata
                    
                    # Save progress after each successful fetch
                    with open(output_file, "w") as f:
                        json.dump(list(fetched_verses.values()), f, indent=2)
                    
                    # Rate limit politeness
                    time.sleep(1.0)
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "throttled" in error_str.lower():
                        wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                        print(f"Rate limited on {ref_str}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"Error fetching {ref_str}: {e}")
                        break
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(references)} references ({len(fetched_verses)} fetched)")

    print(f"\n=== Fetch complete: {len(fetched_verses)} verses saved to {output_file} ===")

def upsert_bible_verses(index, pc):
    """Upsert Bible verses from 'fetched_verses.json' to Pinecone."""
    print("\n--- Upserting Bible Verses to Pinecone ---")
    
    input_file = "fetched_verses.json"
    try:
        with open(input_file, "r") as f:
            verses = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{input_file}' not found. Run fetch_bible_verses first.")
        return

    print(f"Found {len(verses)} verses to upsert.")
    
    batch_size = 10
    
    for i in range(0, len(verses), batch_size):
        batch_verses = verses[i:i + batch_size]
        batch_texts = [v['content'] for v in batch_verses]
        batch_ids = [v['id'] for v in batch_verses]
        
        try:
            embeddings = get_embeddings_batch(batch_texts, pc)
            
            batch_vectors = []
            for j, emb in enumerate(embeddings):
                # Remove 'id' from metadata since it's already the vector id
                meta = {k: v for k, v in batch_verses[j].items() if k != 'id'}
                batch_vectors.append({
                    "id": batch_ids[j],
                    "values": emb,
                    "metadata": meta
                })
            
            index.upsert(vectors=batch_vectors)
            print(f"Upserted batch {i//batch_size + 1}/{(len(verses) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            print(f"Error embedding/upserting batch starting at {i}: {e}")

def main():
    validate_env()
    
    # Initialize clients
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_INDEX_HOST)
    
    # Step 1: Fetch verses from ESV API and save to JSON
    # fetch_bible_verses()
    
    # Step 2: Upsert verses from JSON to Pinecone
    upsert_bible_verses(index, pc)
    
    # Uncomment to also ingest Kindle clippings:
    # ingest_clippings(index, pc)
    
    print("\nIngestion process complete!")

if __name__ == "__main__":
    main()

