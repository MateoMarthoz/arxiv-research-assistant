from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
import jwt
import datetime
from typing import Optional
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from backend.integration import handleUserQuery, index_name, cleanup_index
from backend.Database import create_index, delete_index


# Load environment variables
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Secret key for JWT token encoding
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 600

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Password hashing context (bcrypt)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# connect to MongoDB
client = MongoClient(MONGO_URI)

# Define MongoDB database and collections
db = client["chat_app"] # database name (auto created)
user_collection = db["users"] # collection for user data
chat_log_collection = db["chat_logs"] # collection for chat logs


# FastAPI app initialization
app = FastAPI()

# Pydantic models
class RegisterRequest(BaseModel):
    username: str
    password: str
    confirm_password: str
    
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenData(BaseModel):
    username: Optional[str] = None

class ChatRequest(BaseModel):
    user_message: str

# Function to hash passwords
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# Function to verify passwords
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Function to create JWT tokens
def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + (expires_delta if expires_delta else datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Function to authenticate user
def authenticate_user(username: str, password: str):
    user_doc = user_collection.find_one({"username": username})
    if not user_doc or not verify_password(password, user_doc["password"]):
        return False
    return user_doc


# Function to decode JWT token and verify authentication
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_index: str = payload.get("index_name")
        if username is None or user_index is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "index_name": user_index}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/register")
def register_user(req: RegisterRequest):
    # New validation: ensure the username is provided and is at least 5 characters long
    if not req.username or len(req.username.strip()) < 5:
        raise HTTPException(status_code=400, detail="Username must be at least 5 characters long.")
    
    # New validation: ensure the password is provided and is at least 5 characters long
    if not req.password or len(req.password.strip()) < 5:
        raise HTTPException(status_code=400, detail="Password must be at least 5 characters long.")
    
    # Check if user already exists in MongoDB
    existing_user = user_collection.find_one({"username": req.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists.")

    # Check if the passwords match
    if req.password != req.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")
    
    # Hash the password and store the user in the DB
    hashed_password = hash_password(req.password)
    user_collection.insert_one({
        "username": req.username,
        "password": hashed_password
    })
    return {"message": f"User '{req.username}' registered successfully."}


# Login endpoint (returns JWT token)
@app.post("/login")
def login_user(req: LoginRequest):
    # 1) Query MongoDB
    user_doc = user_collection.find_one({"username": req.username})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # 2) Compare hashed password
    if not verify_password(req.password, user_doc["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # 3) Create a new index for this session.
    user_index_name = create_index()  # create a brand new index

    # 4) Create JWT token
    # After successful login, include index_name in the response or update the session in some way
    access_token = create_access_token(
        data={"sub": req.username, "index_name": user_index_name}
    )

    return {"access_token": access_token, "index_name": user_index_name, "token_type": "bearer"}


# Protected chat endpoint (requires authentication)
@app.post("/chat")
def chat_with_llm(req: ChatRequest, user=Depends(get_current_user)):
    username = user["username"]
    user_index = user["index_name"]

    result = handleUserQuery(req.user_message, user_index)  # pass the index name
    assistant_msg = result["assistant_message"]

    chat_log_collection.update_one(
        {"username": username},
        {"$push": {"chat_history": {"user": req.user_message, "bot": assistant_msg}}},
        upsert=True
    )
    return {"assistant_message": assistant_msg}


@app.on_event("shutdown")
def on_shutdown():
    cleanup_index()
    print("[main] ephemeral index removed.")


# Get chat history endpoint
@app.get("/get_chat_history")
def get_chat_history(username: str = Depends(get_current_user)):
    doc = chat_log_collection.find_one({"username": username})
    if not doc or "chat_history" not in doc:
        return {"chat_history": []}
    return {"chat_history": doc["chat_history"]}

# Clear chat history endpoint
@app.delete("/clear_chat")
def clear_chat(username: str = Depends(get_current_user)):
    result = chat_log_collection.delete_many({"username": username})

    return {"message": "Chat history cleared", "deleted_count": result.deleted_count}
