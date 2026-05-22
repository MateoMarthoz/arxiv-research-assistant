import os
import pytest
from backend.pdf_ingest import ingest_uploaded_pdf

@pytest.fixture
def test_pdf_path():
    # Adjust path to match your real test file
    return "testPaper.pdf"

@pytest.fixture
def pinecone_index_name():
    return "ou7oa7nxqn"

def test_pdf_ingestion_runs(test_pdf_path, pinecone_index_name):
    assert os.path.exists(test_pdf_path), "Test PDF file does not exist"

    num_chunks = ingest_uploaded_pdf(test_pdf_path, pinecone_index_name)

    assert num_chunks > 0, "No chunks were created or stored"
