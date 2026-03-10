from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "I love machine learning",
    "AI is the future",
    "Dogs are great pets"
]

query = "Artificial Intelligence"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

sorted_scores = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)

for index, score in sorted_scores:
    print(documents[index], score)