from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from pinecone import Pinecone


model_name = "sergeyzh/BERTA"
model = HuggingFaceEmbeddings(model_name = model_name)


try:
    pc = Pinecone(
        api_key= st.secrets['PINECONE_API_KEY']
    )
except Exception as e:
    pc = None

index_name = "berta-index"
try:
    index = pc.Index(index_name) if pc else None
except Exception as e:
    index = None


def context(query_text):

    if not pc:
        raise RuntimeError(f"Pinecone doesn't exist")
    if not index:
        raise RuntimeError(f"Index doesn't exist")
            
    query_vector = model.embed_query(query_text)

    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    found_text = []
    for match in results['matches']:
        text = match['metadata'].get('text', '')
        found_text.append(text)


    return found_text
