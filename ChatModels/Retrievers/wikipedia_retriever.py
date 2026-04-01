from dotenv import load_dotenv
from langchain_community.retrievers import WikipediaRetriever

load_dotenv()

import wikipedia
wikipedia.API_URL = "https://en.wikipedia.org/w/api.php"

retriever = WikipediaRetriever(top_k_results=2,lang='en')
query = 'History of India and Pakistan'
docs = retriever.invoke(query)
for i, doc in enumerate(docs):
    print(f"\n --Result {i+1}--")
    print(f"Content :\n {doc.page_content}")
