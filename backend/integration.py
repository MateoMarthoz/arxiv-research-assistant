import os
from time import sleep

from backend.extract_filter import classify_and_extract
from backend.data_utils import fetch_papers, verify_results, chunk_papers, format_query
from backend.Database import create_index, save_embeddings_to_index, delete_index
from backend.retrieval import answer_query, clear_memory
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create the index once (for the current session)
index_name = create_index()
index_obj = pc.Index(index_name)


def handleUserQuery(user_input: str, user_index: str) -> dict:
    """
    This function replicates your notebook's handleUserQuery logic without an infinite loop.

    It first classifies the user input into two actions:
      1) "search": Fetches academic papers from Arxiv using the provided filters,
         verifies the results, embeds the full paper text chunks (with metadata) into Pinecone,
         and then calls the retrieval+LLM pipeline to generate a summary answer.
      2) "discuss": Uses the retrieval+LLM pipeline to generate a discussion answer.

    The returned dictionary always includes an "assistant_message" key that your FastAPI endpoint and UI will display.
    """
    # Instantiate the index object for this session's index.
    index_obj = pc.Index(user_index)

    # Classify the input
    classification = classify_and_extract(user_input)
    print("Classification result:", classification)
    action = classification.get("action", "error")

    # If no valid academic query (i.e. no filters) is detected, use the "discuss" branch.
    if not format_query(classification.get("filters", {})):
        question = classification.get("question", "")
        raw_answer = answer_query(classification, index_obj, top_k=100)
        answer_str = raw_answer.content if hasattr(raw_answer, "content") else str(raw_answer)
        print("Generated Answer (discuss branch):", answer_str)
        return {"action": "discuss", "assistant_message": answer_str}

    # --- SEARCH branch ---
    filters = classification.get("filters", {})
    print("\nRunning SEARCH with filters:", filters)

    # Step 1: Fetch papers from Arxiv
    papers = fetch_papers(filters)
    verified_papers = verify_results(papers, filters)
    print(f"[integration] Found {len(verified_papers)} verified paper(s).")

    # Determine number of papers to return; default to 1 if not provided
    try:
        num_papers = int(filters.get("max_results", 1))
    except ValueError:
        num_papers = 1
    num_papers = min(num_papers, len(verified_papers))

    # Step 2: Chunk papers (retrieve full text chunks with metadata)
    chunks = chunk_papers(verified_papers[:num_papers])
    texts = [chunk.get("text", "") for chunk in chunks]
    # (Optional: you can extract titles, authors, dates if needed for metadata)
    titles = [chunk.get("title", "Unknown Title") for chunk in chunks]
    authors = [chunk.get("author", ["unknown"]) for chunk in chunks]
    dates = [chunk.get("date", "unknown") for chunk in chunks]

    print("Extracted texts:")
    print(texts)

    # Step 3: Embed the full paper text chunks into Pinecone in batches
    if texts:
        batch_size = 200
        for i in range(0, len(texts), batch_size):
            print("Processing batch starting at index:", i)
            text_batch = texts[i:i+batch_size]
            title_batch = titles[i:i+batch_size]
            author_batch = authors[i:i+batch_size]
            date_batch = dates[i:i+batch_size]
            source_batch = ["arxiv"] * len(text_batch)
            save_embeddings_to_index(
                index_name=user_index,
                text=text_batch,
                titles=title_batch,
                authors=author_batch,
                dates=date_batch,
                sources=source_batch,
                namespace="default"
            )
            sleep(1)

    # Finally, generate a response using the retrieval+LLM pipeline
    raw_answer = answer_query(classification, index_obj, top_k=100)
    print("Generated Answer (search branch):", raw_answer)
    answer_str = raw_answer.content if hasattr(raw_answer, "content") else str(raw_answer)

    return {
        "action": "search",
        "assistant_message": answer_str,
        "paper_count": len(verified_papers),
        "message": "Search complete; returning generated answer."
    }

def cleanup_index(user_index):
    """
    Delete the ephemeral Pinecone index for the given user and clear any cached retrieval memory.
    """
    delete_index(user_index)
    clear_memory()
    print(f"[integration] Ephemeral index '{user_index}' has been cleaned up.")