import os
import pickle
import faiss
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


## Load the document:


chunk_size = 500
overlap = 100
all_chunks = []

if not os.path.exists("docs"):
    print("[ERROR] 'docs' folder does not exist.")
    exit()

for file in os.listdir("docs"):
    with open(os.path.join("docs", file), "r") as f:
        document = f.read()


## Split the document into chunks:


    for i in range(0, len(document), chunk_size - overlap):
        chunk = document[i:i + chunk_size]
        all_chunks.append({"chunk": chunk, "file": file})
    
    print(f"Processed {file}, total chunks so far: {len(all_chunks)}")

print(f"Total chunks created: {len(all_chunks)}")


## Generate embeddings for each chunk:

embeddings = []

for items in all_chunks:
    response = client.embeddings.create(
        input = items["chunk"],
        model = "text-embedding-3-small"
    )
    vector = response.data[0].embedding
    embeddings.append(vector)

## Convert the list of embeddings to a numpy array:

embedding_matrix = np.array(embeddings).astype("float32")

## Build the FAISS index:

dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

## Save the index and the chunks:

os.makedirs("vector_db", exist_ok = True)
faiss.write_index(index, "vector_db/faiss_index")

## Save the chunks to a file:

with open("vector_db/chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("FAISS index and chunks saved successfully.")