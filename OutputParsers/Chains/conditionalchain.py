from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import Field, BaseModel
from typing import Literal
from langchain_core.runnables import RunnableLambda, RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'deepseek-ai/DeepSeek-V3.2',
    task="text-generation",
   
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):  # Used PydanticOutputParser because the output is not in our control there is no guarantee that the LLM will give only positive and negative as sentiment, it can give anything else also so to enforce schema and validation
    sentiment : Literal['positive','negative'] = Field(description='Give me the sentiment of the following feedback')

parser1 = PydanticOutputParser(pydantic_object=Feedback)
prompt1 = PromptTemplate(
    template="Classify the sentiment of the following text into positive or negative \n{feedback}\n{format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser1.get_format_instructions()}

)

prompt2 = PromptTemplate(
    template="Write an appropriate response for this positive feedback \n{feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response for this negative feedback \n{feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),  #condition 1 , chain 1
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),  #condition 2 , chain 2
    RunnableLambda(lambda x: "could not find sentiment") # Used Runnable Lambda as there was no default chain to convert that condition into chain
)
classifier_chain = prompt1 | model | parser1  
chain = classifier_chain | branch_chain       
res = chain.invoke({'feedback': 'This is a very wonderful phone'}) #feedback
print(res)



