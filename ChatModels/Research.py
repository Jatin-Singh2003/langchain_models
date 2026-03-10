from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
st.header('Research Tool')

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
   
)

model = ChatHuggingFace(llm=llm)
user_input = st.text_input("Enter Your Prompt:")
if st.button('Summarize'):
    res = model.invoke(user_input)
    st.write(res.content)