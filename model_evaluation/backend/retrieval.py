import os
#from langchain_openai import OpenAIEmbeddings
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings # supposedly the new,correct import
from pinecone import Pinecone
# from integration.Embedder import embed # for script
from backend.Embedder import embed # for notebook
from langchain_community.chat_models import ChatOpenAI # supposedly the new,correct import
#from langchain_openai import ChatOpenAI
#from langchain.chat_models import ChatOpenAI    
from time import sleep
# Initialize the Pinecone client using the API key from environment variables
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Memory buffers to store past interactions and retrieved documents
conversation_history = []  # Stores the last few queries and responses
document_memory = []  # Stores relevant document chunks for future context

def retrieve_chunks(query: str, index_obj, top_k: int = 100):
    tries = 0
    query_response = embed([{"text": query}])
    query_embedding = query_response[0]["values"]
    results = index_obj.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="default"
    ) 
    retrieved_chunks = []
    if "matches" in results and results["matches"]:
        for match in results["matches"]:
            chunk = {
                "text": match["metadata"].get("text", "No text available"),
                "source": match["metadata"].get("source", "Unknown source")
            }
            retrieved_chunks.append(chunk)
            document_memory.append(chunk)
    else:
        print("⚠️ No relevant documents found in Pinecone, wait for 5 seconds and trying again...") 
        sleep(5)
        if tries == 0:
            tries +=1
            results = index_obj.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="default"
            ) 
            retrieved_chunks = []
            if "matches" in results and results["matches"]:
                for match in results["matches"]:
                    chunk = {
                        "text": match["metadata"].get("text", "No text available"),
                        "source": match["metadata"].get("source", "Unknown source")
                    }
                retrieved_chunks.append(chunk)
                document_memory.append(chunk)
                print("After retrying, relevant documents are found")
            else:
                print("⚠️ Still! No relevant documents found in Pinecone.") 
            
    return retrieved_chunks

from transformers import GPT2TokenizerFast
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def truncate_prompt(conversation_part: str, document_part: str, query: str, max_tokens: int = 128000) -> str:
    """
    Constructs a prompt from conversation history, document content, and query.
    
    If the conversation history plus query exceed max_tokens,
    the conversation history will be dropped, preserving only the query.
    
    If the document part exceeds the allowed token count, the last part of the document is preserved.
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
        # This should not happen, but if it does, return a truncated version of the non-document part.
        print("Warning: Non-document part exceeds max tokens even after dropping conversation!")
        truncated_non_doc = tokenizer.decode(non_doc_tokens[:max_tokens])
        return truncated_non_doc
    else:
        if doc_count > allowed_doc_tokens:
            print('Warning: Document part exceeds max token, document length: ', doc_count, ', allowed document tokens: ', allowed_doc_tokens, 'truncating document part')
            # Instead of keeping the first tokens, keep the last allowed_doc_tokens tokens from the document.
            truncated_doc_tokens = doc_tokens[-allowed_doc_tokens:]
            truncated_doc = tokenizer.decode(truncated_doc_tokens)
        else:
            truncated_doc = document_part

    # Combine the non-document part with the (possibly truncated) document part.
    prompt = non_doc_text + "\n" + truncated_doc + "\n"
    return prompt



def prepare_prompt(query: str, retrieved_chunks: list):
    # Retrieve conversation history
    history_text = "\n".join(conversation_history[-10:])
    conversation_part = f"Conversation history:\n{history_text}\n\n" if history_text else ""
    
    # Optionally filter by paper title if mentioned in the query
    paper_title = None
    # if "typiclust" in query.lower():
    #     paper_title = "TypiClust"  # or extract a more precise title if needed

    if paper_title:
        filtered_chunks = [chunk for chunk in document_memory if paper_title.lower() in chunk.get("paper_title", "").lower()]
    else:
        filtered_chunks = document_memory

    memory_text = "\n\n".join([chunk["text"] for chunk in filtered_chunks])
    document_part = f"Relevant documents:\n{memory_text}\n\n" if memory_text else ""
    
    prompt = (
        f"{conversation_part}"
        f"{document_part}"
        f"Query: {query}\nAnswer:"
    )
    truncate_prompt(conversation_part, document_part, query)

    tokens = tokenizer.encode(prompt) 
    
    return prompt



import os 

def call_llm(): 
    experiment_group = os.getenv("EXPERIMENT_GROUP", "balanced").lower()
    
    if experiment_group == "conservative":
        temperature = 0.2
        top_p = 1.0
        max_tokens = 1000
        frequency_penalty = 0.0
        presence_penalty = 0.0
    elif experiment_group == "balanced":
        temperature = 0.5
        top_p = 0.9
        max_tokens = 1200
        frequency_penalty = 0.1
        presence_penalty = 0.1
    elif experiment_group == "open":
        temperature = 0.7
        top_p = 0.8
        max_tokens = 1500
        frequency_penalty = 0.0
        presence_penalty = 0.0 

    print(f"Using experiment group: {experiment_group}")
    print(f"Parameters: temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}, frequency_penalty={frequency_penalty}, presence_penalty={presence_penalty}")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    return llm


def answer_query(query: str, index_obj, top_k: int = 100, use_memory: bool = False):
    if use_memory:
        # Combine previously stored document memory with fresh retrieval
        new_retrieval = retrieve_chunks(query, index_obj, top_k)
        retrieved_chunks = document_memory + new_retrieval
    else:
        retrieved_chunks = retrieve_chunks(query, index_obj, top_k)
    
    prompt = prepare_prompt(query, retrieved_chunks)
    llm = call_llm()
    answer = llm.invoke(prompt)
    conversation_history.append(f"Query: {query}\nAnswer: {answer}")
    return answer

def clear_memory():
    conversation_history.clear()
    document_memory.clear()
    print("Memory cleared.")
