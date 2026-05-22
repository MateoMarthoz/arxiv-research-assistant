import json
import pytest
from io import BytesIO

# Import functions from your gradio_app module
from gradio_app import (
    login_user,
    register_user,
    submit_message,
    logout_user,
    load_chat_history,
    after_login,
    process_uploaded_pdf,
)
# Also import ingest_uploaded_pdf so we can monkeypatch it in process_uploaded_pdf
from backend.pdf_ingest import ingest_uploaded_pdf

# --- Helper Classes and Functions ---

class FakeResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

def fake_post_login_success(url, json):
    # Simulate a successful login response
    if "/login" in url:
        return FakeResponse(200, {"access_token": "FAKE_TOKEN", "index_name": "my_index"})
    # For registration simulation
    elif "/register" in url:
        return FakeResponse(200, {"message": f"User '{json.get('username')}' registered successfully."})
    # For chat submission simulation
    elif "/chat" in url:
        return FakeResponse(200, {"assistant_message": "Fake assistant reply"})
    return FakeResponse(404, {})

def fake_get_history(url, headers):
    # Simulate an empty chat history
    return FakeResponse(200, {"chat_history": []})

def fake_delete_clear(url, headers):
    # Simulate a clear chat successful deletion
    return FakeResponse(200, {"message": "Chat history cleared", "deleted_count": 1})

# Dummy file-like object for PDF uploads (only needs a 'name' attribute)
class DummyFile:
    def __init__(self, name):
        self.name = name

# --- Tests for Frontend Functions ---

def test_login_user_success(monkeypatch):
    # Monkeypatch requests.post so that login_user sees a success response.
    monkeypatch.setattr("gradio_app.requests.post", lambda url, json: fake_post_login_success(url, json))
    session = {}
    message, success = login_user("testuser", "password123", session)
    assert success is True
    assert "Welcome" in message
    assert session.get("access_token") == "FAKE_TOKEN"
    assert session.get("index_name") == "my_index"

def test_register_user_success(monkeypatch):
    monkeypatch.setattr("gradio_app.requests.post", lambda url, json: fake_post_login_success(url, json))
    message = register_user("newuser", "password123", "password123")
    assert "registered successfully" in message

def test_submit_message_success(monkeypatch):
    # Monkeypatch requests.post for the chat endpoint
    monkeypatch.setattr("gradio_app.requests.post", lambda url, json, headers: fake_post_login_success(url, json))
    # Prepare a fake session that indicates the user is logged in
    session = {
        "logged_in": True,
        "access_token": "FAKE_TOKEN",
    }
    chat_history = []
    new_input = "Hello there"
    returned_input, updated_history, _ = submit_message(new_input, chat_history, session)
    # Check that the function returns an empty string (clearing the text box) and updates the history
    assert returned_input == ""
    assert len(updated_history) == 1
    assert updated_history[0] == (new_input, "Fake assistant reply")

def test_logout_user():
    session = {"logged_in": True, "access_token": "SOME_TOKEN"}
    message, logged_in_status = logout_user(session)
    assert logged_in_status is False
    assert session["access_token"] is None
    assert "logged out" in message

def test_load_chat_history(monkeypatch):
    # Monkeypatch requests.get for chat history
    monkeypatch.setattr("gradio_app.requests.get", lambda url, headers: fake_get_history(url, headers))
    # Prepare fake session with a token and logged-in state
    session = {"logged_in": True, "access_token": "FAKE_TOKEN"}
    history = load_chat_history(session)
    assert isinstance(history, list)
    assert len(history) == 0

def test_after_login(monkeypatch):
    # Test that after_login combines login and chat history retrieval
    monkeypatch.setattr("gradio_app.requests.post", lambda url, json: fake_post_login_success(url, json))
    monkeypatch.setattr("gradio_app.requests.get", lambda url, headers: fake_get_history(url, headers))
    session = {}
    msg, success, history = after_login("testuser", "password123", session)
    assert success is True
    assert "Welcome" in msg
    # For our fake history, we expect an empty list
    assert history == []


def test_process_uploaded_pdf_not_logged_in():
    # If the user is not logged in, process_uploaded_pdf should return a message accordingly.
    dummy_pdf = DummyFile("dummy.pdf")
    session = {"logged_in": False}
    result = process_uploaded_pdf(dummy_pdf, session)
    assert "must be logged in" in result