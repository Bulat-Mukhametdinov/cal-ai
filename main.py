import streamlit as st
import langchain_groq
from utils import *

model = langchain_groq.ChatGroq(
    model_name = 'deepseek-r1-distill-llama-70b',
    api_key =st.secrets['GROQ_API_KEY'],
    )

st.title("hei")

if prompt := st.chat_input("What is up"):
    ans = model_answer(model, prompt)

# st.button('Reset Chat', on_click=reset_conversation)