from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    
)

model = ChatHuggingFace(llm=llm)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

doc1 = Document(
    page_content='Cristiano Ronaldo is the best football player in the world with over 960 goals scored, known for his commitment and work ethic',
    metadata = {'team': "Real Madrid"}

)
doc2 = Document(
    page_content= 'Lionel Messi is a very all round footballer who has many tremendous skills and god like abilities',
    metadata = {'team': "Barcelona"}

)
doc3 = Document(
     page_content='Neymar is regarded as the prince of modern day football and is a awesome dribbler',
     metadata = {'team': "Barcelona"}

)
doc4 = Document(
    page_content= 'Kevin de bryne is regarded as one of the most unique midefielders of all time',
    metadata = {'team': "Manchester City"}

)
doc5 = Document(
    page_content='Kylian Mbappe is one of the best young youth players who is known for his goalscoring abilities ',
    metadata = {'team': 'PSG'}
)
docs = [doc1,doc2,doc3,doc4,doc5]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory= 'chroma_db',
    collection_name='sample'
)

#add documents
vector_store.add_documents(docs)

#view documents
print(vector_store.get(include=['embeddings','documents','metadatas']))

#search documents
print(vector_store.similarity_search(
    query='Who among them is highest goalscorer in football?',
    k=2
)) 

#search with similarity score
print(vector_store.similarity_search_with_score(
    query="Who among them is a great dribbler?",
    k=1
))

#meta data filtering
print(vector_store.similarity_search_with_score(
    query='Barcelona players',
    filter={'team': 'Barcelona'}
))