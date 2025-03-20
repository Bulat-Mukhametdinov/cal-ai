import streamlit as st
from extra_streamlit_components import CookieManager 
import langchain_groq
import json
from utils import *
cookie_manager = CookieManager()

cookies = {}
while not cookies or "user_id" not in cookies:
    cookies = cookie_manager.get_all()
    if not cookies or "user_id" not in cookies:
        st.warning("Cookies are being loaded. Please wait...")
        # Simulate a small delay to avoid busy-waiting
        time.sleep(0.5)

cookies = cookie_manager.get_all()
st.write(cookies)
model = langchain_groq.ChatGroq(
    model_name = 'deepseek-r1-distill-llama-70b',
    api_key =st.secrets['GROQ_API_KEY'],
    )

if "chats" not in st.session_state:
    st.session_state.chats = {}  # Dictionary to store chat histories
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# if "user_id" not in cookies:
#     USER_ID = str(os.urandom(8).hex())  # Generate a unique user ID
#     cookie_manager.set("user_id", USER_ID)  # Save the user ID in cookies
# else:
USER_ID = cookies["user_id"]

st.write(f"Ваш уникальный ID: {USER_ID}")
CHAT_HISTORY_FILE = os.path.join(os.getcwd(), f"chat_history_{USER_ID}.json")

# Step 3: Load chat history from file on startup
# Load chat history from file on startup
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
new_chat_name = st.sidebar.text_input("Enter chat name (optional)", placeholder="New chat name")
if st.sidebar.button("Add Chat"):
    chat_name = new_chat_name.strip() or f"Chat {len(st.session_state.chats) + 1}"
    print(st.session_state.chats)
    if chat_name not in st.session_state.chats:
        st.session_state.chats[chat_name] = []  # Initialize new chat history
        st.session_state.current_chat = chat_name  # Switch to the new chat
        save_chats()  # Save updated chat history to file
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

#Section 3: Delete Current Chat
if st.sidebar.button("Delete Current Chat"):
    if st.session_state.current_chat:
        del st.session_state.chats[st.session_state.current_chat]
        st.session_state.current_chat = None
        save_chats()
        st.rerun()

#Step 5: Main chat interface
st.title("AI Chat App")

if st.session_state.current_chat:
    print(st.session_state.chats)
    st.subheader(f"Chat: {st.session_state.current_chat}")


    # Display chat history
    for message in st.session_state.chats[st.session_state.current_chat]:
        with st.chat_message(message["role"]):
            st.write(message['content'])

    if prompt := st.chat_input("What is up"):
        ans = model_answer(model, prompt)
        save_chats()

