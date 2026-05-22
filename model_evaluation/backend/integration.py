import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# from Search_or_Discuss import classify_and_extract
from Model_Evaluation.Search_or_Discuss import classify_and_extract
from Model_Evaluation.data_utils import fetch_papers, verify_results, chunk_papers
from Model_Evaluation.Database import create_index, save_embeddings_to_index, delete_index
from Model_Evaluation.retrieval import answer_query, clear_memory
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = create_index()
index_obj = pc.Index(index_name)

def handleUserQuery(user_input: str) -> dict:
    """
    This function replicates your notebook's handleUserQuery logic without an infinite loop.
    
    It classifies the user input into two actions:
      1) "search": It fetches academic papers from Arxiv using the provided filters,
         verifies the results, and then returns a formatted string that lists the titles
         and abstracts of the top N papers (where N is specified by filters' "max_results",
         or defaults to 1 if not provided).
         
      2) "discuss": It calls the retrieval + LLM pipeline to generate a discussion answer,
         converts the answer to a plain string, and returns it.
         
    The returned dictionary always includes an "assistant_message" key that your FastAPI
    endpoint and UI will display.
    """
    classification = classify_and_extract(user_input)
    action = classification.get("action", "error")
    
    if action == "search":
        filters = classification.get("filters", {})
        print(f"[integration] SEARCH with filters: {filters}")
        
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
        
        # Format the title and abstract for each of the top papers
        formatted_results = []
        for i in range(num_papers):
            paper = verified_papers[i]
            # Format: "Paper X: Title: ... Abstract: ..."
            formatted_results.append(
                f"Paper {i+1}:\nTitle: {paper.title}\nAbstract: {paper.summary}"
            )
        
        result_text = "\n\n".join(formatted_results)
        
        #embed the paper texts for retrieval later:
        #for paper in verified_papers[:num_papers]:
             # For embedding purposes, use the full text if available (or abstract)
        #     save_embeddings_to_index(index_name, [paper.summary])
        

                # Embed full paper chunks instead of just abstracts
        
        ######################################################################
        ##### This is where you would embed the full paper text chunks #######
        ######################################################################

        chunks = chunk_papers(verified_papers[:num_papers])  # Get full paper content as chunks
        texts = [chunk["text"] for chunk in chunks]  # Extract just the text

        #TODO：does it work
        if texts:
            # Save the full paper text chunks to the index
            batch_size = 200
            for i in range(0, len(texts), batch_size):
                text_batch = texts[i:i+batch_size] 
                save_embeddings_to_index(index_name=user_index, text=text_batch)
            sleep(1)

        #TODO: answer search question
        return {
            "action": "search",
            "assistant_message": result_text,
            "paper_count": len(verified_papers),
            "message": "Search complete; returning top papers with titles and abstracts."
        }
    
    elif action == "discuss":
        question = classification.get("question", "")
        print(f"[integration] DISCUSS question: {question}")
        
        # Run the retrieval + LLM generation pipeline
        raw_answer = answer_query(question, index_obj, top_k=3)
        print("[integration] Generated LLM answer:", raw_answer)
        
        # Convert answer to a plain string in case it's an AIMessage or custom object
        if hasattr(raw_answer, "content"):
            answer_str = raw_answer.content
        else:
            answer_str = str(raw_answer)
            
        return {
            "action": "discuss",
            "assistant_message": answer_str
        }
    
    else:
        print("[integration] Action not recognized or invalid JSON from classification.")
        return {
            "action": "error",
            "assistant_message": "Action not recognized by the pipeline.",
            "message": "Please try again."
        }

def cleanup_index():
    """
    Called from main.py on shutdown (or as needed) to delete the ephemeral Pinecone index
    and clear any cached retrieval memory.
    """
    delete_index(index_name)
    clear_memory()
    print(f"[integration] Ephemeral index '{index_name}' has been cleaned up.")
