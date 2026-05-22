import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import random
from types import SimpleNamespace
import backend.Database  

# Fake Pinecone client that simulates the behavior of Pinecone.
class FakePinecone:
    def __init__(self):
        self.indexes = []          # List to hold index names.
        self.created_indexes = []  # Log of create_index calls.
    
    def list_indexes(self):
        return [{"name": name} for name in self.indexes]
    
    def create_index(self, name, dimension, metric, spec):
        self.indexes.append(name)
        self.created_indexes.append((name, dimension, metric, spec))
    
    def delete_index(self, index_name):
        if index_name in self.indexes:
            self.indexes.remove(index_name)
        else:
            # Simulate behavior when deletion fails.
            raise Exception("Index does not exist")
    def Index(self, index_name):
        # Return a fake index instance that captures upsert calls.
        return FakeIndex(index_name)
    
class FakeIndex:
    def __init__(self, name):
        self.name = name
        self.upsert_called = False
        self.upsert_vectors = None
        self.upsert_namespace = None
    
    def upsert(self, vectors, namespace):
        self.upsert_called = True
        self.upsert_vectors = vectors
        self.upsert_namespace = namespace

# fake embed function that simulates embedding by returning the length of the text.
def fake_embed(input):
    return [{"values": [len(item["text"])]} for item in input]

@pytest.fixture
def fake_pc(monkeypatch):
    fake = FakePinecone()
    monkeypatch.setattr(backend.Database, "pc", fake)
    return fake

def test_create_index(fake_pc, monkeypatch):
    # Override random.choice to always return "x" so the generated index name is predictable.
    monkeypatch.setattr(random, "choice", lambda seq: "x")
    
    index_name = backend.Database.create_index()
    # Expected index name is 10 "x" characters.
    assert index_name == "xxxxxxxxxx"
    # Verify the index is added to the fake client's indexes.
    assert "xxxxxxxxxx" in fake_pc.indexes
    # Also, check that create_index logged the call.
    assert any(item[0] == "xxxxxxxxxx" for item in fake_pc.created_indexes)

def test_delete_index(fake_pc, capsys):
    # Add an index to delete.
    fake_pc.indexes.append("testindex")
    backend.Database.delete_index("testindex")
    # After deletion, the index should no longer be in the fake client's indexes.
    assert "testindex" not in fake_pc.indexes

    # Test deleting a index that does not exist
    backend.Database.delete_index("nonexistent")
    captured = capsys.readouterr().out
    assert "does not exist" in captured  # Check that an error message was printed.

def test_save_embeddings_to_index(fake_pc, monkeypatch):
    # Create fake input data.
    texts = ["text1", "text2"]
    titles = ["Title1", "Title2"]
    authors = [["Author1"], ["Author2"]]
    dates = ["2020", "2021"]
    sources = ["source1", "source2"]

    # Create a fake index object by calling the Index method on fake_pc.
    fake_index = fake_pc.Index("fake_index")
    
    monkeypatch.setattr(fake_pc, "Index", lambda index_name: fake_index)
    monkeypatch.setattr(backend.Database,"embed", fake_embed)
    
    backend.Database.save_embeddings_to_index("fake_index", texts, titles, authors, dates, sources)
    
    assert fake_index.upsert_called is True
    assert fake_index.upsert_namespace == "default"

    # Verify that the vectors list has the expected length.
    assert len(fake_index.upsert_vectors) == 2

    vectors = fake_index.upsert_vectors
    
    expected_text = f"'{texts[0]}'\ntext from {titles[0]}\nauthored by {', '.join(authors[0])}"
    expected_title = titles[0].lower()
    expected_embedding = [len(expected_text)]  
    
    assert vectors[0]["metadata"]["text"] == expected_text
    assert vectors[0]["metadata"]["title"] == expected_title
    assert vectors[0]["values"] == expected_embedding
