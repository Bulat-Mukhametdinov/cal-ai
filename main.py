import langsmith
import langsmith
import streamlit as st
from utils import *
from agents import AgentAnswerPipeline
from streamlit_cookies_controller import CookieController
import torch
from dotenv import load_dotenv


### INITIALIZATIONS BEGIN ###
torch.classes.__path__ = [] # dirty fix - add this line to manually set it to empty.

# langsmith tracing integration
langsmith_env = ["LANGCHAIN_TRACING_V2", "LANGCHAIN_ENDPOINT", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"]
if all([param in st.secrets for param in langsmith_env]):
    load_dotenv(dotenv_path=".streamlit/secrets.toml", override=True)
    print(f"LangSmith tracing enabled: {langsmith.utils.tracing_is_enabled()}")
else:
    print("LangSmith environment parameters not found.")

agent = AgentAnswerPipeline() # ReAct based agent for answering

controller = CookieController()
cookies = {}
cookies = controller.getAll()

# If cookies was not already loaded
if "user_id" not in cookies:
    print('waiting for cookies..')
    time.sleep(2)
    cookies = controller.getAll()

if "user_id" not in cookies:
    USER_ID = str(os.urandom(8).hex())  # Generate a unique user ID
    controller.set("user_id", USER_ID)  # Save the user ID in cookies
else:
    USER_ID = cookies["user_id"]

st.write(f"Your unique identifier: {USER_ID}")
CHAT_HISTORY_FILE = os.path.join(os.getcwd(), f"chats/chat_history_{USER_ID}.json")

if "chats" not in st.session_state:
    st.session_state.chats = load_chats(CHAT_HISTORY_FILE)
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
else:
    chat_history = st.session_state.chats.get(st.session_state.current_chat, [])
    agent.init_chat_history(chat_history)

### INITIALIZATION END ###

### SIDEBAR BEGIN ###

st.sidebar.title("Chat Sessions")

# Create New Chat
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

# List of Existing Chats
st.sidebar.subheader("Your Chats")
for chat_name in st.session_state.chats.keys():
    if st.sidebar.button(chat_name, key=f"chat_button_{chat_name}"):
        st.session_state.current_chat = chat_name  # Switch to the selected chat

# Add another divider before delete button
st.sidebar.markdown("---")

# Delete Current Chat
if st.sidebar.button("Delete Current Chat"):
    if st.session_state.current_chat:
        del st.session_state.chats[st.session_state.current_chat]
        st.session_state.current_chat = None
        save_chats(CHAT_HISTORY_FILE)
        st.rerun()

### SIDEBAR END ###

### MAIN CHAT INTERFACE BEGIN ###

st.title("AI Chat App")

if st.session_state.current_chat:
    st.subheader(f"Chat: {st.session_state.current_chat}")

    # Display chat history
    for message in st.session_state.chats[st.session_state.current_chat]:
        message = message.model_dump()
        with st.chat_message(message["type"]):
            st.write(message['content'])

    if prompt := st.chat_input("What is up"):
        # Showing up user's entered messge
        with st.chat_message("user"):
            st.write(prompt)
        
        # Answer for user
        ans = agent(prompt)
        
        # Showing up llm's answer
        with st.chat_message('assistant'):
            st.write(ans)
        
        # Chat saving
        st.session_state.chats[st.session_state.current_chat] = agent.get_chat()
        save_chats(CHAT_HISTORY_FILE)
