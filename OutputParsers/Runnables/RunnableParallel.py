from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'deepseek-ai/DeepSeek-V3.2',
    task="text-generation",
   
)

model = ChatHuggingFace(llm=llm)
prompt1 = PromptTemplate(
    template= 'Generate a tweet about  {topic}',
    input_variables= ['topic']
)
prompt2 = PromptTemplate(
    template= 'Generate a linkedin post about {topic}',
    input_variables=['topic']
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
     'tweet': RunnableSequence(prompt1,model,parser),
     'linkedin': RunnableSequence(prompt2,model,parser)
 })

res = parallel_chain.invoke({'topic': "AI"})
print(res)
