import streamlit as st
from langchain_groq import ChatGroq

MODEL_NAME = "deepseek-r1-distill-qwen-32b"
llm = ChatGroq (
    model=MODEL_NAME, 
    api_key=st.secrets["GROQ_API_KEY"],
)

PROOF_MODEL = "deepseek-r1-distill-llama-70b"
llm_proof = ChatGroq(
    model=PROOF_MODEL,
    api_key=st.secrets["GROQ_API_KEY"],
)


if __name__ == "__main__":
    print(llm_proof.invoke("Who are you?").content)