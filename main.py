import streamlit as st
import langchain_groq
import json
from utils import *
from llm import llm
from agents import AgentAnswerPipeline
import random
import torch

torch.classes.__path__ = [] # add this line to manually set it to empty.
agent = AgentAnswerPipeline()
CHAT_HISTORY_FILE = "chats_data.json"


if "chats" not in st.session_state:
    st.session_state.chats = load_chats(CHAT_HISTORY_FILE)
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

st.sidebar.title("Chat Sessions")

# Section 1: Create New Chat
if st.sidebar.button("Create New Chat"):
    new_chat_name = ''
    flag = True
    while flag or new_chat_name in st.session_state.chats:
        new_chat_name = generate_chat_name()
        flag = False
    
    st.session_state.chats[new_chat_name] = []  # Initialize new chat history
    st.session_state.current_chat = new_chat_name  # Switch to the new chat

# Add a divider between sections
st.sidebar.markdown("---")

# Section 2: List of Existing Chats
st.sidebar.subheader("Your Chats")
for chat_name in st.session_state.chats.keys():
    if st.sidebar.button(chat_name, key=f"chat_button_{chat_name}"):
        st.session_state.current_chat = chat_name  # Switch to the selected chat
        agent.init_chat_history(st.session_state.chats[chat_name])

# Add another divider before delete button
st.sidebar.markdown("---")

# Section 3: Delete Current Chat
if st.sidebar.button("Delete Current Chat"):
    if st.session_state.current_chat:
        del st.session_state.chats[st.session_state.current_chat]
        st.session_state.current_chat = None
        save_chats()
        st.rerun()

# Step 5: Main chat interface
st.title("AI Chat App")

if st.session_state.current_chat:
    st.subheader(f"Chat: {st.session_state.current_chat}")


    # Display chat history
    print(st.session_state.chats)
    print(st.session_state.current_chat)
    for message in st.session_state.chats[st.session_state.current_chat]:
        message = message.model_dump()
        with st.chat_message(message["type"]):
            st.write(message['content'])

    if prompt := st.chat_input("What is up"):
        with st.chat_message("user"):
            st.write(prompt)
        
        ans = agent(prompt)
        
        with st.chat_message('assistant'):
            st.write(ans)
        
        st.session_state.chats[st.session_state.current_chat] = agent.get_chat()
        save_chats(CHAT_HISTORY_FILE)