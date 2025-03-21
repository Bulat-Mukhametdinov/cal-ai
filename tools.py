import streamlit as st
from rag import context
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain.agents import load_tools




class GetContext(BaseModel):
    query_text: str = Field(description="Text for searching in retrieval data base.")
    

@tool(args_schema=GetContext)
def get_context(query_text: str) -> str:
    """Return context data from calculus textbook"""
    answer_list = context(query_text)
    return "\n".join(answer_list)

tools = [
    get_context,
    load_tools(["wolfram-alpha"], wolfram_alpha_appid=st.secrets['WOLFRAM_ALPHA_APPID'])[0]
]