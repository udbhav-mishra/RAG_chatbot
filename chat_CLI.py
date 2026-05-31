
import os
import faiss
import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv


## Load api key and initialize OpenAI client:


try:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")
    
    client = OpenAI(api_key = api_key)
except Exception as e:
    print(f"Error loading API key: {e}")
    exit()
    

## Load the FAISS index and chunks:


if not os.path.exists("vector_db/faiss_index"):
    print("[ERROR] FAISS index not found.")
    print("Run build.py first.")
    exit()

index = faiss.read_index("vector_db/faiss_index")

try:
    with open("vector_db/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
except Exception as e:
    print(f"[ERROR] Failed to load chunks.pkl: {e}")
    exit()


## User query:


query = input("Enter your query: ")


## Create embedding for the query:


response = client.embeddings.create(model = "text-embedding-3-small", input = query)
query_vector = np.array(response.data[0].embedding).astype("float32")


## search similar chunks in the index:


k = 5
try:
    distances, indices = index.search(np.array([query_vector]), k)    
    retrived_chunks = [chunks[i] for i in indices[0]]
    context = "\n".join(
        [f"Source: {chunk['file']}\n{chunk['chunk']}" for chunk in retrived_chunks]
    )
except Exception as e:
    print(f"[ERROR] Failed to search FAISS index: {e}")
    exit()


## Generate response using the retrieved context:


chat_response = client.chat.completions.create(
    model = "gpt-4.1",
    messages = [
        {"role": "system", "content": "Answer using provided context only. If you don't know the answer, say you don't know."},
        {"role": "user", "content": f"""Context: {context}\n\nQuestion: {query}"""}
    ]
)

answer = chat_response.choices[0].message.content
print("\n Retrieved Context:\n", context)
print("\n Answer:\n", answer)