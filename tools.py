import streamlit as st
from rag import context
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field

class GetContext(BaseModel):
    query_text: str = Field(description="Text for searching in retrieval data base.")
    

@tool(args_schema=GetContext)
def get_context(query_text: str) -> str:
    """Return context data from calculus textbook"""
    answer_list = context(query_text)
    return "\n".join(answer_list)



tools = [
    get_context,
]