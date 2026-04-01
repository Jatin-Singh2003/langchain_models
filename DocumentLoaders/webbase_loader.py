from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
   
)

model = ChatHuggingFace(llm=llm)
url ='https://www.flipkart.com/apple-iphone-17-sage-256-gb/p/itmcfa57eff7729c?pid=MOBHFN6YNAG4ZTHS&lid=LSTMOBHFN6YNAG4ZTHSWUQQUI&marketplace=FLIPKART&q=iphone+17&store=tyy%2F4io&srno=s_1_1&otracker=AS_QueryStore_OrganicAutoSuggest_1_7_na_na_na&otracker1=AS_QueryStore_OrganicAutoSuggest_1_7_na_na_na&fm=organic&iid=ec8ab994-8406-4241-b5ef-a56dd0c2cac8.MOBHFN6YNAG4ZTHS.SEARCH&ppt=clp&ppn=iphone-premium-feb-26-store&ssid=x21mrq9kds0000001774545310653&qH=c9eeb2d6cc488f0b&ov_redirect=true'
loader = WebBaseLoader(url)
docs = loader.load()
parser = StrOutputParser()
prompt = PromptTemplate(
    template= 'Answer the {question} for the following text \n {text}',
    input_variables=['text','question']
)
chain = prompt | model | parser
res = chain.invoke({'question': 'Tell me about a few features of the phone?', 'text': docs[0].page_content})
print(res)