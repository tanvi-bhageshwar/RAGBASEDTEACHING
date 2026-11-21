import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests


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

def inference(prompt,model="llama3.2"):
    r=requests.post("http://localhost:11434/api/generate",json=
                    {
                        # "model":"deepseek-r1",
                        "model":"llama3.2",
                        "prompt":prompt,
                        "stream":False
                      })
    response=r.json()
    print(response)
    return response


df=joblib.load('embeddings.joblib')

# taking question and converting it into question_embedding
incoming_query=input("Ask a Question: ")
question_embedding=create_embedding([incoming_query])[0]
# print(question_embedding)
# FInd similarities of question_embedding iwth other embeddings

# using numpy to convert our embeddings into 2D data
#flatten is used to make the arr plane
similarities=cosine_similarity(np.stack(df['embedding']),[question_embedding]).flatten()
# print(similarities)
top_results=5
max_indx=similarities.argsort()[::-1][0:top_results]
print(max_indx)
new_df=df.loc[max_indx]
 # to get top 3 indices from max to min
# print(new_df[["title","number","text"]])

prompt=f''' I am teaching web development in my sigma web developmen course. Here are video subtitle chunks contaning video title,video number,
start time in second, end time in seconds,
the text at that time:
{new_df[["title","number","start","end","text"]].to_json(orient="records")}
-----------------------
"{incoming_query}"
User asked this question related to video chunks, you have to answer in a human way(dont mention the above format, it is just for u ) where and 
how much content is taught in which video (in which video and at what timestamp and guide use to go to
that particular video. if user asks unrelated question, tell him that you can only answer
questions related to the course
'''

with open("prompt.txt","w") as f:
    f.write(prompt)

response=inference(prompt)["response"]
print(response)

with open("response.txt","w") as f:
    f.write(response)
# for index,item in new_df.iterrows():
#     print(index,item["title"],item["number"],item["text"],item["start"],item["end"])