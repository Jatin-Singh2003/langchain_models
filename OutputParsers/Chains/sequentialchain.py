from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    task="text-generation",
   
)

model = ChatHuggingFace(llm=llm)
# 1st prompt -> detailed report
prompt1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
)


# 2nd Prompt -> Summary in 5 lines
prompt2 = PromptTemplate(
    template = 'Generate a 5 pointer summary for the following text.\n{text}',
    input_variables = ['text']
)
parser = StrOutputParser()
chain = prompt1 | model | parser | prompt2 | model | parser
res = chain.invoke({'topic': 'Umemployment in India'})
print(res)
chain.get_graph().print_ascii() 