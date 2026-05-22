import pytest
from fastapi.testclient import TestClient
from app.main import app, user_collection, chat_log_collection, index_name
import os

# Create a TestClient instance for our FastAPI app
client = TestClient(app)

@pytest.fixture(autouse=True)
def clear_db():
    """
    Fixture to clean up the MongoDB collections before each test,
    so that tests run in isolation.
    """
    user_collection.delete_many({})
    chat_log_collection.delete_many({})
    yield
    user_collection.delete_many({})
    chat_log_collection.delete_many({})

def test_register_success():
    # Test valid registration
    payload = {
        "username": "testuser",
        "password": "password123",
        "confirm_password": "password123"
    }
    response = client.post("/register", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert f"User '{payload['username']}' registered successfully." in data["message"]

def test_register_short_username():
    # Test registration with a username that is too short
    payload = {
        "username": "usr",
        "password": "password123",
        "confirm_password": "password123"
    }
    response = client.post("/register", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "Username must be at least 5 characters long" in data["detail"]

def test_register_passwords_do_not_match():
    # Test registration with mismatched passwords
    payload = {
        "username": "testuser",
        "password": "password123",
        "confirm_password": "different"
    }
    response = client.post("/register", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "Passwords do not match" in data["detail"]

def test_login_invalid_user():
    # Test login with non-existing user
    payload = {
        "username": "nonexistent",
        "password": "password123"
    }
    response = client.post("/login", json=payload)
    assert response.status_code == 401
    data = response.json()
    assert "Invalid username or password" in data["detail"]

def test_register_and_login():
    # First register a new user, then login successfully
    reg_payload = {
        "username": "testuser",
        "password": "password123",
        "confirm_password": "password123"
    }
    reg_resp = client.post("/register", json=reg_payload)
    assert reg_resp.status_code == 200

    login_payload = {"username": "testuser", "password": "password123"}
    login_resp = client.post("/login", json=login_payload)
    assert login_resp.status_code == 200
    data = login_resp.json()
    # Check that an access token is returned and also the index name is returned
    assert "access_token" in data
    assert data.get("index_name") == index_name

def test_chat_endpoint_without_token():
    # Test that accessing chat endpoint without a valid token fails
    payload = {"user_message": "Hello"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 401  # Unauthorized

def test_chat_endpoint_with_token():
    # Register and login a user, then test protected chat endpoint
    reg_payload = {
        "username": "testuser",
        "password": "password123",
        "confirm_password": "password123"
    }
    client.post("/register", json=reg_payload)

    login_payload = {"username": "testuser", "password": "password123"}
    login_resp = client.post("/login", json=login_payload)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Test sending a message to the /chat endpoint
    chat_payload = {"user_message": "Hello"}
    response = client.post("/chat", json=chat_payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "assistant_message" in data

def test_get_chat_history_empty():
    # Test that new user has an empty chat history.
    reg_payload = {
        "username": "testuser",
        "password": "password123",
        "confirm_password": "password123"
    }
    client.post("/register", json=reg_payload)
    login_payload = {"username": "testuser", "password": "password123"}
    login_resp = client.post("/login", json=login_payload)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/get_chat_history", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["chat_history"] == []

def test_clear_chat_history():
    # Register, login, simulate a chat, and then clear the chat history.
    reg_payload = {
        "username": "testuser",
        "password": "password123",
        "confirm_password": "password123"
    }
    client.post("/register", json=reg_payload)
    login_payload = {"username": "testuser", "password": "password123"}
    login_resp = client.post("/login", json=login_payload)
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Add a message to the chat history via the /chat endpoint
    client.post("/chat", json={"user_message": "Hello"}, headers=headers)
    
    # Now clear the chat history
    clear_resp = client.delete("/clear_chat", headers=headers)
    assert clear_resp.status_code == 200
    data = clear_resp.json()
    assert "Chat history cleared" in data["message"]
    # Confirm that chat_log_collection is actually cleared 