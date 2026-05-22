from langchain_community.document_loaders import PyPDFLoader
from backend.Database import save_embeddings_to_index
from langchain.text_splitter import RecursiveCharacterTextSplitter

"""
Description:
This script processes user-uploaded PDFs by loading the file, splitting it into text chunks, 
and storing the embedded results into a Pinecone index. It uses LangChain for PDF loading 
and text splitting, and a custom save_embeddings_to_index function to upload the data.
"""


def chunk_documents(documents, chunk_size=500, overlap=100):
    """
    Accepts LangChain Documents (e.g., from PyPDFLoader),
    and returns text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunks = splitter.split_documents(documents)
    return chunks

def ingest_uploaded_pdf(file_path: str, user_index: str):
    """
    Loads, chunks, and stores an uploaded PDF.
    """
    print(f"Loading uploaded PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    print("Chunking uploaded document...")
    chunks = chunk_documents(documents)  # reuse the same chunking logic

    texts = []
    titles = []
    authors = []
    dates = []
    sources = []

    filename = file_path.split("/")[-1]  # or use os.path.basename(file_path)

    for i, chunk in enumerate(chunks):
        texts.append("\n from user's uploaded PDF file: " + chunk.page_content)
        titles.append(filename)             # or "Uploaded PDF"
        authors.append([])           # no author info in uploaded PDFs
        dates.append("00000000")                # static or parsed if you want
        sources.append("uploaded_pdf")      # to distinguish from 'arxiv'

    print(f"Storing {len(chunks)} chunks to index: {user_index}")
    
    save_embeddings_to_index(
        index_name=user_index,
        text=texts,
        titles=titles,
        authors=authors,
        dates=dates,
        sources=sources
    )
    
    return len(chunks)
