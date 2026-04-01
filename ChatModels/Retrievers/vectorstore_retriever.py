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
documents = [
    Document(page_content='Cristiano Ronaldo is the best football player in the world with over 960 goals scored, known for his commitment and work ethic'),
    Document(page_content= 'Lionel Messi is a very all round footballer who has many tremendous skills and god like abilities'),
    Document(page_content='Neymar is regarded as the prince of modern day football and is a awesome dribbler'),
    Document(page_content= 'Kevin de bryne is regarded as one of the most unique midefielders of all time'),
    Document(page_content='Kylian Mbappe is one of the best young youth players who is known for his goalscoring abilities')
    
]
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    collection_name='ny_collection'
)

retriever = vectorstore.as_retriever(search_kwargs = {'k': 2})
query = 'Who is the best dribbler in football?'
res = retriever.invoke(query)
for i, doc in enumerate(res):
    print(f"\n--Result{i+1}--")
    print(doc.page_content)