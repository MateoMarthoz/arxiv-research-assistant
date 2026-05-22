from openai import OpenAI
from dotenv import load_dotenv
import os

"""
Description:
This script uses OpenAI's API to generate embeddings for input text using the "text-embedding-3-small" model.
"""

# Load environment variables
load_dotenv()

# Create OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(data):

    # Create a list of texts from the input data.
    text_inputs = [d['text'] for d in data] if isinstance(data, list) else [data]

    # Get embeddings for the texts using 'text-embedding-3-small' from OpenAI.
    response = client.embeddings.create(
        input=text_inputs,
        model="text-embedding-3-small"
    )

    # Return the embeddings in a simple list of dictionaries.
    return [{"values": item.embedding} for item in response.data]



