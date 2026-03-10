from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2' )
document = [
    'Cristiano Ronaldo is the best football player in the world with over 960 goals scored known for his commitment and work ethic',
    'Lionel Messi is a very all round footballer who has many tremendous skills and god like abilities',
    'Neymar is regarded as the prince of modern day football and is a awesome dribbler'
    'Kevin de bryne is regarded as one of the most unique midefielders of all time'
    'Kylian Mbappe is one of the most talented new gen players with many records'
    ]
query = 'Tell me about Cristiano Ronaldo'
doc_emb = embeddings.embed_documents(document)
query_emb = embeddings.embed_query(query)

scores = cosine_similarity([query_emb], doc_emb)[0]

index, score = sorted(list(enumerate(scores)), key = lambda x:x[1])
print(query)
print(document[index])
print("Similarity score is:" ,score)
