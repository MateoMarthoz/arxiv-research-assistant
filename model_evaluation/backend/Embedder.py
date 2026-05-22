import os
from dotenv import load_dotenv
load_dotenv()

USE_OPENAI = os.getenv("USE_OPENAI_EMBEDDINGS", "True").lower() == "true"

if USE_OPENAI:
    from openai import OpenAI
    MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def embed(data):
        # data 可以是字符串或包含字典的列表
        text_inputs = [d['text'] for d in data] if isinstance(data, list) else [data]
        response = client.embeddings.create(
            input=text_inputs,
            model=MODEL_NAME
        )
        return [{"values": item.embedding} for item in response.data]
else:
    from sentence_transformers import SentenceTransformer
    hf_mini_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def embed(text):
        if not isinstance(text, list):
            text = [text]
        embeddings = hf_mini_model.encode(text).tolist()
        return [{"values": emb} for emb in embeddings]
