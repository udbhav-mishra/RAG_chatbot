import os
import faiss
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel


## Load api key and initialize OpenAI client:


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = api_key)


## Security scheme for API key authentication (if needed):


VALID_API_KEY = set(os.getenv("VALID_API_KEY", "").split(","))  # Set of valid API keys
security = HTTPBearer()
print(f"Valid API keys: {VALID_API_KEY}")  # Debugging statement

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = credentials.credentials
    print("Received:", credentials.credentials)
    print("Allowed:", VALID_API_KEY)
    if api_key not in VALID_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key   


## Load the FAISS index and chunks:


index = faiss.read_index("vector_db/faiss_index")
with open("vector_db/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


## User query:


class Query(BaseModel):
    query: str

app = FastAPI(title="RAG API")
@app.get("/")
def home():
    return {"message": "Welcome to the RAG API. Use the /query endpoint to ask questions."}

@app.post("/query")

def answer_query(query: Query, api_key: str = Depends(verify_api_key)):
    query = query.query

    try:
        # Create embedding for the query
        response = client.embeddings.create(model = "text-embedding-3-small", input = query)
        query_vector = np.array(response.data[0].embedding).astype("float32")


        # search FAISS index for similar chunks


        k = 5
        distances, indices = index.search(np.array([query_vector]), k)
        retrived_chunks = [chunks[i] for i in indices[0]]
        context = "\n".join(
            [f"Source: {chunk['file']}\n{chunk['chunk']}" for chunk in retrived_chunks]
        )


        # Generate response using the retrieved context


        chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        chat_response = client.chat.completions.create(
            model = chat_model,
            messages = [
                {"role": "system", "content": "Answer using provided context only. If you don't know the answer, say you don't know."},
                {"role": "user", "content": f"""Context: {context}\n\nQuestion: {query}"""}
            ]
        )

        return {"answer": chat_response.choices[0].message.content}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
