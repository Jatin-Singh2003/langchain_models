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
prompt = PromptTemplate(
    template= "Give me five facts about {topic}",
    input_variables=['topic']
)


parser = StrOutputParser()
chain = prompt | model | parser
res = chain.invoke({'topic': 'Soccer'})
print(res)
chain.get_graph().print_ascii()