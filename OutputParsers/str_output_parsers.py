from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    task="text-generation",
   
)

model = ChatHuggingFace(llm=llm)
# 1st prompt -> detailed report
template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
)


# 2nd Prompt -> Summary in 5 lines
template2 = PromptTemplate(
    template = 'Write a 5 line summary on the following text./n{text}',
    input_variables = ['text']
)
prompt1 = template1.invoke({'topic': 'blackhole'})
res = model.invoke(prompt1)
prompt2 = template2.invoke({'text': res.content})
res1 = model.invoke(prompt2)
print(res1.content)