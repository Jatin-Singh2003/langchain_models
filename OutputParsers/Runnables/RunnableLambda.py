from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    task="text-generation",
   
)

model = ChatHuggingFace(llm=llm)
def word_counter(text):
    return len(text.split())
runnable_word_counter = RunnableLambda(word_counter)

prompt = PromptTemplate( 
    template= 'Write a joke about {topic}',
    input_variables= ['topic']
)
parser = StrOutputParser()
joke_gen_chain = RunnableSequence(prompt,model,parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_counter)
})
final_chain = RunnableSequence(joke_gen_chain,parallel_chain)
res = final_chain.invoke({'topic': 'Donald Trump'})
final_res = """ {} \n Word count - {}""".format(res['joke'],res['word_count'])
print(final_res)