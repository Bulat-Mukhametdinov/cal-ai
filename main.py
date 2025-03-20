import streamlit as st
import langchain_groq
import json
from utils import *

model = langchain_groq.ChatGroq(
    model_name = 'deepseek-r1-distill-llama-70b',
    api_key =st.secrets['GROQ_API_KEY'],
    )

if "chats" not in st.session_state:
    st.session_state.chats = {}  # Dictionary to store chat histories
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

CHAT_HISTORY_FILE = "chats_data.json"

# Step 3: Load chat history from file on startup
def load_chats():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}
    return {}

st.session_state.chats = load_chats()

# Step 4: Save chat history to file
def save_chats():
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(st.session_state.chats, file, ensure_ascii=False, indent=4)

st.sidebar.title("Chat Sessions")

# Section 1: Create New Chat
st.sidebar.subheader("Create New Chat")
new_chat_name = st.sidebar.text_input("Enter chat name", placeholder="New chat name")
if st.sidebar.button("Add Chat") and new_chat_name.strip():
    if new_chat_name not in st.session_state.chats:
        st.session_state.chats[new_chat_name] = []  # Initialize new chat history
        st.session_state.current_chat = new_chat_name  # Switch to the new chat
        save_chats()
    else:
        st.sidebar.warning("Chat name already exists!")

# Add a divider between sections
st.sidebar.markdown("---")

# Section 2: List of Existing Chats
st.sidebar.subheader("Your Chats")
for chat_name in st.session_state.chats.keys():
    if st.sidebar.button(chat_name, key=f"chat_button_{chat_name}"):
        st.session_state.current_chat = chat_name  # Switch to the selected chat

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
    for message in st.session_state.chats[st.session_state.current_chat]:
        with st.chat_message(message["role"]):
            st.write(message['content'])

    if prompt := st.chat_input("What is up"):
        ans = model_answer(model, prompt)
        save_chats()

st.markdown(f"The remote ip is {get_remote_ip()}")
print(f"The remote ip is {get_remote_ip()}")