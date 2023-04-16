import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Load the fine-tuned model
model = SentenceTransformer('prajjwal1/bert-tiny')

# Load the dataset
data = pd.read_csv('/Users/seansullivan/Downloads/output2.csv')
questions = data['Question'].tolist()

questions = data['Question'].tolist()
answers = data['Answer'].tolist()

# Embed the questions
question_embeddings = model.encode(questions)

# Create a Faiss index
faiss_index = faiss.IndexFlatL2(question_embeddings.shape[1])
faiss_index.add(np.array(question_embeddings).astype('float32'))

def search(query, top_k=7):
    query_embedding = model.encode([query])[0]  # Embed the user's query
    distances, indices = faiss_index.search(np.array([query_embedding]), top_k)  # Search for the top_k closest embeddings

    # Retrieve the corresponding questions and answers for the closest matches
    results = []
    for distance, idx in zip(distances[0], indices[0]):
        question = questions[idx]
        answer = answers[idx]
        results.append((question, answer, distance))

    return results

user_query = "how can I charge people to view my quizlet flashcard set"
top_k = 5

results = search(user_query, top_k)

print(f"Top {top_k} results for the query: {user_query}")
for i, (question, answer, distance) in enumerate(results):
    print(f"{i+1}. Question: {question}\n   Answer: {answer}\n   Distance: {distance}\n")
