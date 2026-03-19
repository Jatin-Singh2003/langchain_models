from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    task="text-generation",
   
)

model = ChatHuggingFace(llm=llm1)


llm2 = HuggingFaceEndpoint(
    repo_id = 'deepseek-ai/DeepSeek-V3.2',
    task="text-generation",
   
)
model1 = ChatHuggingFace(llm = llm2)

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following text \n {text}",
    input_variables = ['text']
)
prompt2 = PromptTemplate(
    template = "Generate 5 question answer pairs from the following text\n{text}",
    input_variables = ['text']

)
prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz in a single document \n notes->{notes} and quiz -> {quiz}",
    input_variables = ['quiz','notes']

)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz': prompt2 | model1 | parser
})
merge_chain = prompt3 | model | parser
chain = parallel_chain | merge_chain
text = """LangChain is an open-source framework designed to help developers build applications powered by large language models (LLMs). It provides tools and abstractions that make it easier to connect language models with external data sources, APIs, and application logic. Instead of calling an LLM directly with a simple prompt, developers can create structured workflows where prompts, models, and output processing are combined into reusable pipelines called chains.

One of the main strengths of LangChain is its modular architecture. It includes components such as prompt templates, models, output parsers, retrievers, and memory. Prompt templates help structure inputs to the model, while output parsers convert raw LLM responses into structured formats like JSON or Python objects. Retrievers allow applications to fetch relevant information from databases or documents, enabling techniques such as retrieval-augmented generation (RAG), where the model answers questions using external knowledge sources.

LangChain also supports advanced workflows using runnables, which allow developers to compose complex pipelines. For example, tasks can be executed sequentially or in parallel, enabling applications to perform multiple LLM calls simultaneously and combine the results. This flexibility makes it useful for building chatbots, document analysis tools, AI assistants, and automated research systems.

Another important feature of LangChain is its integration ecosystem. It works with many LLM providers, including models hosted on platforms like OpenAI and Hugging Face. It also integrates with vector databases and cloud services, allowing developers to build scalable AI applications. Because of its extensibility and strong community support, LangChain has become one of the most widely used frameworks for developing production-grade LLM applications."""
res = chain.invoke({'text': text})

print(res)
