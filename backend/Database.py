from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from backend.Embedder import embed
import random
import string
import uuid

"""
This script provides utility functions to:
- Create and delete Pinecone vector indexes
- Convert and embed text data using a custom embedder
- Save embeddings along with metadata (title, author, date, source) to Pinecone

"""

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def create_index(dimension=1536, metric="cosine", cloud="aws", region="us-east-1"):
    """
    this function created a vector index to access pinecone
    """
    # Generate random name for an index
    str_characters = string.ascii_lowercase + string.digits
    index_name = ''.join(random.choice(str_characters) for _ in range(10))

    # Check if the index already exists
    if index_name not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            ) 
        )
        print(f"Index '{index_name}' created successfully!")
    else:
        print(f"Index '{index_name}' already exists, skipping creation.")
    
    # Return the name of the index created
    return index_name

def delete_index(index_name):
    """
    this function deletes designated pinecone vector index, because the maximum limit for existing vector index is 5
    """
    try:
        pc.delete_index(index_name)
        print(f"Index '{index_name}' deleted successfully!")
    except Exception as e: # if the index does not exist catch the error
        print(f"Index '{index_name}' does not exist or was already deleted.")


def to_int(s: str) -> int:
    return int(s)

def save_embeddings_to_index(index_name, text, titles, authors, dates, sources, namespace="default"):
    # Retrieve the index instance
    index = pc.Index(index_name)
    data_with_ids = []

    for t, ti, a, d, s in zip(text, titles, authors, dates, sources):
        # Normalize the title and authors to lowercase
        normalized_title = ti.lower()
        normalized_authors = [x.lower() for x in a] if a else []
        # Merge the text chunk with title and author details
        t = f"'{t}'\ntext from {ti}\nauthored by {', '.join(a)}"
        data_with_ids.append({
            "id": str(uuid.uuid4()),
            "text": t,
            "title": normalized_title,
            "author": normalized_authors,  # Save lowercased authors
            "date": to_int(d),
            "source": s
        })

    print('pinecone gonna store: ')
    for i in data_with_ids[:5]:
        print(i)

    # Embed the data
    embeddings = embed(data_with_ids)

    # Prepare vectors for upsert with metadata
    vectors = []
    for d, e in zip(data_with_ids, embeddings):
        vectors.append({
            "id": d['id'],
            "values": e['values'],
            "metadata": {
                "text": d['text'],
                "title": d['title'],
                "author": d["author"], 
                "date": d["date"],
                "source": d['source']
            }
        })
    
    # Upsert embeddings into the index
    index.upsert(
        vectors=vectors,
        namespace=namespace
    )
    print(f"Upserted {len(vectors)} embeddings into index '{index_name}', namespace '{namespace}'.")
