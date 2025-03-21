import streamlit as st
from rag import context
from langchain.tools import tool
from pydantic import BaseModel, Field
from llm import llm_proof
from langchain.agents import load_tools


### RAG search Tool begin ###

class GetContext(BaseModel):
    query_text: str = Field(description="Text for searching in retrieval data base.")


@tool(args_schema=GetContext)
def get_context(query_text: str) -> str:
    """Return context data from calculus textbook"""
    answer_list = context(query_text)
    return "\n".join(answer_list)

### RAG search Tool end ###


### Proof Check Tool begin ###

prompt = """You are best at checking proof correctness
Check this proof
{proof}
Return your short opinion on this proof"""


class ProofCheck(BaseModel):
    proof: str = Field(description="Proof that needs to be checked.")


@tool(args_schema=ProofCheck)
def proof_check(proof: str) -> str:
    """Returns the correctness of proof"""
    try:
        correctness = llm_proof.invoke(prompt.format(proof = proof)).content.split('</think>')[-1]
        return correctness
    except Exception as e:
        print("Error on checking proof")
        print(e)
        return "Error on checking proof"

### Proof Check Tool end ###

tools = [
    get_context,
    proof_check,
    load_tools(["wolfram-alpha"], wolfram_alpha_appid=st.secrets['WOLFRAM_ALPHA_APPID'])[0],
]