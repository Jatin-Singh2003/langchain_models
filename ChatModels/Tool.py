
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
st.header('Research Tool')
paper_input = st.selectbox("Select Research Paper Name",["BERT","GPT-3","GAN's"])
style_input = st.selectbox("Select Explanation Style",["Beginner-Friendly","Technical",'Code-oriented','Mathematical'])
length_input = st.selectbox("Select Length",["Short","Medium","Long"])
#template
template = PromptTemplate(
    template = """ 
    
     Please Summarize the research paper titled '{paper_input}' with the following
     specifications:
     Explanation Style: '{style_input}'
     Explanation Length: '{length_input}'
         1. Mathematical Details:
         Include relevant mathematical equations if present in the paper.
         Explain the mathematical concepts using simple, intuitive code snippets where applicable.
         2. Analogies
        use relatable analogies to simplify complex ideas.
        If certain information is not available in the paper, respond with: "Insufficient Information available" instead of guessing
        Ensure the summary is clear, accurate, and aligned with the provided style and length.""",
        input_variables = ['paper_input','style_input','length_input']

)

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
   
)

model = ChatHuggingFace(llm=llm)
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
     })
if st.button('Summarize'):
    res = model.invoke(prompt)
    st.write(res.content)