import os
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
from backend.Embedder import embed
from langchain_community.chat_models import ChatOpenAI
from time import sleep


"""
Description:
This script implements a retrieval-based question answering system using Pinecone and an LLM. It:
- Embeds text with OpenAIEmbeddings.
- Retrieves document chunks from Pinecone with metadata filters.
- Manages token limits via GPT2TokenizerFast.
- Builds prompts from conversation history, retrieved documents, and user queries.
- Generates responses with ChatOpenAI.
- Maintains conversation history.

Usage:
- retrieve_chunks(): Fetch relevant documents.
- prepare_prompt(): Construct the query prompt.
- answer_query(): Generate an answer.
- clear_memory(): Reset conversation history.
"""


# Initialize the Pinecone client using the API key from environment variables
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

conversation_history = []  # Stores the last 10 queries and responses 

def parse_date(date_str: str) -> int:
    """
    turn YYYYMMDD or YYYYMMDDHHmm string into integer 
    """
    return int(date_str)


def retrieve_chunks(classification, index_obj, top_k: int = 100):
    tries = 0
    query = classification['question']
    query_response = embed([{"text": query}])
    query_embedding = query_response[0]["values"]

    # create metadata filter for pinecone retrieval
    filter_condition = {}

    # filter for author
    author_filter = classification["filters"].get("author")
    if author_filter is not None and str(author_filter).lower() != "null":
        # check if the target author is in the list of authors
        filter_condition["author"] = {"$in": [author_filter.lower()]}

    # do range query for the date metadata using start_date and end_date 
    start_date = classification["filters"].get("start_date")
    end_date = classification["filters"].get("end_date")
    date_filter = {}
    if start_date is not None and str(start_date).lower() != "null":
        date_filter["$gte"] = parse_date(start_date)
    if end_date is not None and str(end_date).lower() != "null":
        date_filter["$lte"] = parse_date(end_date)
    if date_filter: 
        filter_condition["date"] = date_filter
 
    print('Filter condition: ')
    print(filter_condition) 

    results = index_obj.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="default",
        filter=filter_condition if filter_condition else None
    )
 
    # parse retrieved results into text chunks
    retrieved_chunks = []
    if "matches" in results and results["matches"]:
        for match in results["matches"]:
            metadata = match["metadata"]
            chunk = {
                "text": metadata.get("text", "No text available"),
                "title": metadata.get("title", "Unknown title"),
                "author": metadata.get("author", "Unknown author"),
                "date": metadata.get("date", "Unknown date"),
                "source": metadata.get("source", "Unknown source")
            }
            retrieved_chunks.append(chunk) 
            
    else:
        # Pinecone may have a cold start delay. If no data is returned, wait 5 seconds and retry.
        print("⚠️ No relevant documents found in Pinecone, wait for 5 seconds and trying again...") 
        sleep(5)
        
        if tries == 0:
            # Ensure only one retry to avoid infinite loop on cold start
            tries +=1
            results = index_obj.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="default",
            filter=filter_condition if filter_condition else None
        )

        retrieved_chunks = []
        if "matches" in results and results["matches"]:
            for match in results["matches"]:
                metadata = match["metadata"]
                chunk = {
                    "text": metadata.get("text", "No text available"),
                    "title": metadata.get("title", "Unknown title"),
                    "author": metadata.get("author", "Unknown author"),
                    "date": metadata.get("date", "Unknown date"),
                    "source": metadata.get("source", "Unknown source")
                }
                retrieved_chunks.append(chunk) 
                print("After retrying, relevant documents are found")
        else:
            print("⚠️ Still! No relevant documents found in Pinecone.") 

    print("Retrieved chunks: ")  
    print(retrieved_chunks)
    return retrieved_chunks

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def truncate_prompt(conversation_part: str, document_part: str, query: str, max_tokens: int = 128000) -> str:
    """
    This function is to check if the prompt length exceeds max token limit

    If the conversation history plus query exceed max_tokens,
    the conversation history will be dropped, preserving only the query.
    
    If the document part exceeds the allowed token count, the latest part of the document is preserved.
    """
    # Construct the non-document part: conversation history + "Query: {query}\nAnswer:"
    non_doc_text = conversation_part + f"Query: {query}\nAnswer:"
    non_doc_tokens = tokenizer.encode(non_doc_text)
    
    # If non-document part exceeds max_tokens, drop conversation history and use only the query prompt.
    query_only_text = f"Query: {query}\nAnswer:"
    query_only_tokens = tokenizer.encode(query_only_text)
    if len(non_doc_tokens) > max_tokens:
        print("Warning: Conversation and query part exceed max tokens! Dropping conversation history.")
        non_doc_text = query_only_text
        non_doc_tokens = query_only_tokens

    non_doc_count = len(non_doc_tokens)
    
    # Calculate the allowed tokens for the document part.
    allowed_doc_tokens = max_tokens - non_doc_count

    # Encode the document part tokens.
    doc_tokens = tokenizer.encode(document_part)
    doc_count = len(doc_tokens)
    
    if allowed_doc_tokens < 0:
        # Prevent user entering long query maliciously, by returning a truncated version of the non-document part.
        print("Warning: Non-document part exceeds max tokens even after dropping conversation!")
        truncated_non_doc = tokenizer.decode(non_doc_tokens[:max_tokens])
        return truncated_non_doc
    else:
        if doc_count > allowed_doc_tokens:
            print('Warning: Document part exceeds max token, document length: ', doc_count, ', allowed document tokens: ', allowed_doc_tokens, 'truncating document part')
            # keep the latest allowed_doc_tokens tokens from the document.
            truncated_doc_tokens = doc_tokens[-allowed_doc_tokens:]
            truncated_doc = tokenizer.decode(truncated_doc_tokens)
        else:
            truncated_doc = document_part

    # Combine the non-document part with the (possibly truncated) document part.
    prompt = non_doc_text + "\n" + truncated_doc + "\n"
    return prompt



def prepare_prompt(classification, retrieved_chunks: list):
    """
    This function is for combining conversation history, retrieved paper chunks and user query into a text prompt
    """
    # Retrieve conversation history
    history_text = "\n".join(conversation_history[-10:])
    conversation_part = f"Conversation history:\n{history_text}\n\n" if history_text else ""
    query = classification['question']

    # retrieve paper chunks and metadata
    memory_text = "\n\n".join([
        f"Paper Title: {chunk.get('title', 'Unknown title')}\n"
        f"Author: {chunk.get('author', 'Unknown author')}\n"
        f"Date: {chunk.get('date', 'Unknown date')}\n"
        f"Text Chunk: {chunk.get('text', 'No text available')}"
        for chunk in retrieved_chunks
    ])
    document_part = f"Documents Provided:\n{memory_text}\n\n" if memory_text else "No Relevant documents found, maybe user did not enter correct paper information\n\n"
    
    prompt = (
        f"{conversation_part}"
        f"{document_part}"
        f"Query: {query}\nAnswer:(When answering do not provide metadata)\n"
    )
    print("\nPrompt passed to LLM:")
    print(prompt)
    truncate_prompt(conversation_part, document_part, query)

    tokens = tokenizer.encode(prompt) 
    
    return prompt

def answer_query(classification, index_obj, top_k: int = 100, use_memory: bool = False):
    """
    function for retrieving paper chunks from pinecone, and passing formatted text prompt into LLM
    """
    query = classification["question"]
    retrieved_chunks = retrieve_chunks(classification, index_obj, top_k)
    
    prompt = prepare_prompt(classification, retrieved_chunks) 
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature = 0.5,
        top_p = 0.9,
        max_tokens = 1200,
        frequency_penalty = 0.1,
        presence_penalty = 0.1)
    answer = llm.invoke(prompt)
    conversation_history.append(f"User Inputed: {query}\nAnswer: {answer}")
    return answer


def clear_memory():
    # clear the user conversation history when user exit
    conversation_history.clear() 
    print("Memory cleared.")
