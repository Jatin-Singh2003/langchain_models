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
loader = TextLoader(
    "/Users/macmini/Desktop/langchain_models/DocumentLoaders/math.txt",
    encoding="utf-8"
)
docs = loader.load()
parser = StrOutputParser()
prompt = PromptTemplate(
    template= 'Write a summary for the following text \n {text}',
    input_variables=['text']
)
chain = prompt | model | parser
res = chain.invoke({'text': docs[0].page_content})
print(res)