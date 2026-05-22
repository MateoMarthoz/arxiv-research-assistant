import pytest
from backend.retrieval import (
    parse_date,
    retrieve_chunks,
    truncate_prompt,
    prepare_prompt,
    answer_query,
    clear_memory,
    conversation_history,
)

# ---------------------------
# Create Fake Dependencies
# ---------------------------

# Fake embed function: returns a fake embedding for each input text.
def fake_embed(data):
    # Create a fake embedding of 1536 elements (all 0.1)
    fake_embedding = [0.1] * 1536
    # Expected output: list of dicts with key "values"
    return [{"values": fake_embedding} for _ in data]

# Fake index object that simulates Pinecone's query and describe_index_stats methods.
class FakeIndex:
    def query(self, vector, top_k, include_metadata, namespace, filter=None):
        # Return a fake query result with one match.
        return {
            "matches": [
                {
                    "metadata": {
                        "text": "Fake document text about machine learning.",
                        "title": "Fake ML Paper",
                        "author": "John Doe",
                        "date": "20250101",
                        "source": "Fake Source"
                    }
                }
            ]
        }
    
    def describe_index_stats(self):
        return {
            "namespaces": {
                "default": {
                    "vector_count": 1
                }
            }
        }

# Fixture to supply our fake index
@pytest.fixture
def fake_index():
    return FakeIndex()

# Override the embed function in retrieval.py with our fake_embed.
@pytest.fixture(autouse=True)
def override_embed(monkeypatch):
    monkeypatch.setattr("backend.retrieval.embed", fake_embed)

# Clear global memory before each test.
@pytest.fixture(autouse=True)
def clear_globals():
    conversation_history.clear()

# ---------------------------
# Begin Tests
# ---------------------------

def test_parse_date():
    # Test that a date string is correctly parsed to an integer.
    assert parse_date("20250101") == 20250101
    assert isinstance(parse_date("20250101"), int)

def test_retrieve_chunks(fake_index):
    # Create a fake classification input with an empty filter.
    classification = {"question": "What is machine learning?", "filters": {}}
    chunks = retrieve_chunks(classification, fake_index, top_k=1)
    # Ensure that we get one chunk back.
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    # Verify that the retrieved text contains our fake text.
    assert "Fake document text" in chunks[0]["text"]

def test_truncate_prompt():
    # Define dummy conversation and document parts.
    conversation_part = "Conversation history:\nQuery: What is ML?\nAnswer: ML is machine learning."
    document_part = "Relevant documents:\nFake document text about machine learning."
    query = "Explain ML in detail."
    # Call truncate_prompt with a high max_tokens so no truncation happens.
    prompt = truncate_prompt(conversation_part, document_part, query, max_tokens=1000)
    # Check that the prompt includes the query.
    assert "Explain ML in detail." in prompt

def test_prepare_prompt():
    # Set up dummy conversation history and document memory.
    conversation_history.extend([
        "Query: What is ML?\nAnswer: ML stands for machine learning."
    ])
    retrieved_chunks = [
        {
            "text": "Fake document text about machine learning.",
            "title": "Fake ML Paper",
            "author": "John Doe",
            "date": "20250101",
            "source": "Fake Source"
        }
    ]
    classification = {"question": "Explain machine learning."}
    prompt = prepare_prompt(classification, retrieved_chunks)
    # Ensure that both conversation history and document info appear in the prompt.
    assert "Conversation history:" in prompt
    assert "Documents Provided:" in prompt
    assert "Fake ML Paper" in prompt

def test_answer_query(monkeypatch, fake_index):
    # Monkeypatch ChatOpenAI.invoke to return a fixed string for testing.
    def fake_invoke(self, prompt, *args, **kwargs):
        return "Simulated answer based on prompt: " + prompt

    monkeypatch.setattr("backend.retrieval.ChatOpenAI.invoke", fake_invoke)
    
    # Create a fake classification input.
    classification = {"question": "What is ML?", "filters": {}}
    answer = answer_query(classification, fake_index, top_k=1)
    # Verify that the simulated answer is returned.
    assert "Simulated answer" in answer
    # Check that the conversation history was updated.
    assert len(conversation_history) == 1
    assert "What is ML?" in conversation_history[0]

def test_clear_memory():
    # Add dummy data to conversation_history.
    conversation_history.extend(["Dummy conversation"])
    clear_memory()
    # Check that conversation_history is empty.
    assert len(conversation_history) == 0
