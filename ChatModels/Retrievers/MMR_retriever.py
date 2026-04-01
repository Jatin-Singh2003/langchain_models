from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    
)

model = ChatHuggingFace(llm=llm)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs= [
    Document(page_content='Machine Learning is a subset of artificial intelligence that focuses on building systems that learn from data and improve over time without explicit programming'),
    
    Document(page_content='Deep Learning is a specialized field within machine learning that uses neural networks with multiple layers to model complex patterns in data'),
    
    Document(page_content='Natural Language Processing enables machines to understand, interpret, and generate human language in a meaningful way'),
    
    Document(page_content='Computer Vision is a field of AI that allows machines to interpret and make decisions based on visual data like images and videos'),
    
    Document(page_content='Reinforcement Learning is a type of machine learning where agents learn to make decisions by interacting with an environment and receiving rewards'),
    
    Document(page_content='Supervised Learning is a machine learning approach where models are trained on labeled data to predict outcomes'),
    
    Document(page_content='Unsupervised Learning involves finding hidden patterns or structures in data without labeled outputs'),
    
    Document(page_content='Generative AI focuses on creating new content such as text, images, and audio using models like transformers and diffusion models'),
    
    Document(page_content='Large Language Models are AI systems trained on massive datasets to understand and generate human-like text'),
    
    Document(page_content='Transfer Learning allows a model trained on one task to be reused for another related task, improving efficiency and performance')
]
vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding

)
retriever = vectorstore.as_retriever(
    search_type = 'mmr',
    search_kwargs = {'k': 3, 'lambda_mult': 0.5}

)
query = 'What are different types of AI?'
res = retriever.invoke(query)
for i, doc in enumerate(res):
    print(f"\n--Result{i+1}--")
    print(doc.page_content)