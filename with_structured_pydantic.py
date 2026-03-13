from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)


class Review(BaseModel):
    summary: str = Field(description="A brief summary of the review")
    sentiment: str = Field(description="positive, negative or neutral sentiment")


structured_model = model.with_structured_output(
    Review,
    method="json_mode"
)

res = structured_model.invoke(
"""The hardware is great but the software feels bloated.
There are too many pre-installed apps that make the phone slow."""
)

print(res)