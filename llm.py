import streamlit as st
from langchain_groq import ChatGroq

MODEL_NAME = "llama3-70b-8192"

llm = ChatGroq (
    model=MODEL_NAME, 
    api_key=st.secrets["GROQ_API_KEY"],
)


if __name__ == "__main__":
    print(llm.invoke("Who are you?").content)