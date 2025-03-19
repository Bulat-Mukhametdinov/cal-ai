from langchain_community.vectorstores import FAISS
import os
from langchain_huggingface import HuggingFaceEmbeddings


model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

db = os.path.join(os.curdir, "db") 

if not os.path.exists(db):
    raise FileNotFoundError(f"The directory {db} does not exist. Please check the path.")

vector_store = FAISS.load_local(db, model, allow_dangerous_deserialization=True)


def context(query):

    retriever = vector_store.similarity_search(query, k=5)

    results = [doc.page_content for doc in retriever]
    return results
