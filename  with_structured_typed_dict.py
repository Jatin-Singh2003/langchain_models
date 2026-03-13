from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import Annotated, TypedDict
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational",
   
)

model = ChatHuggingFace(llm=llm)
class Review(TypedDict):
    summary : Annotated[str, "A brief summary of the review"]
    sentiment : Annotated[str, "Return the sentiment of the review either negative,positive or neutral"]

structured_model = model.with_structured_output(Review, method='json_mode')
res = structured_model.invoke(""" The hardware is great but the software feels bloated.
There are too many pre-installed apps that make the system and the phone slow and affects the performance.""")
print(res)
