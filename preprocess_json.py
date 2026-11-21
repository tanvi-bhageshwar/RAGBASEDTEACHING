
import requests
import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
# Function to create embeddings
def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        },
        timeout=120
    )

    # Print actual error if request fails
    if r.status_code != 200:
        print("Status Code:", r.status_code)
        print("Error:", r.text)
        return None

    embedding = r.json()["embeddings"]
    return embedding


# List all JSON files
jsons = os.listdir("jsons")

my_dicts = []
chunk_id = 0

# Process JSON files
for json_file in jsons:

    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)
    print(f"Creating embeddings for {json_file}")
    chunks = content["chunks"]

    # Extract text from chunks
    texts = [c["text"] for c in chunks]

    # Generate embeddings in batches
    batch_size = 8
    embeddings = []

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        batch_embeddings = create_embedding(batch)

        if batch_embeddings is None:
            raise Exception("Embedding generation failed")

        embeddings.extend(batch_embeddings)

    # Add chunk_id and embedding to each chunk
    for i, chunk in enumerate(chunks):

        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        
        chunk_id += 1

        my_dicts.append(chunk)
        
        # print(chunk)

   


# Convert dictionary to DataFrame
df = pd.DataFrame.from_records(my_dicts)
#Save the dataframe using joblib
# joblib is mainly used for Machine Learning Models , but we can use ithere also
joblib.dump(df,'embeddings.joblib')
# print(df)

# Optional: save DataFrame
# df.to_csv("chunks_with_embeddings.csv", index=False)
