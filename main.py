from audio_recorder_streamlit import audio_recorder
from streamlit_float import *
from sound import *
import streamlit as st
import langsmith
import langsmith
import langsmith
import streamlit as st
from utils import *
from agents import AgentAnswerPipeline
from streamlit_cookies_controller import CookieController
import torch
import os
from llm import llm

### INITIALIZATIONS BEGIN ###
torch.classes.__path__ = [] # dirty fix - add this line to manually set it to empty.

# langsmith tracing integration
langsmith_env = ["LANGCHAIN_TRACING_V2", "LANGCHAIN_ENDPOINT", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"]
if all([param in st.secrets for param in langsmith_env]):
    if os.environ.get("LANGCHAIN_API_KEY") is None:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
        os.write(1, f"LangSmith tracing enabled: {langsmith.utils.tracing_is_enabled()}\n".encode("ASCII"))
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

st.write(f'<span style="color: gray; opacity: 0.5;">Yor unique identifier is "{USER_ID}"</span>', unsafe_allow_html=True)
CHAT_HISTORY_FILE = os.path.join(os.getcwd(), f"chats/chat_history_{USER_ID}.json")

if "chats" not in st.session_state:
    st.session_state.chats = load_chats(CHAT_HISTORY_FILE)
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
else:
    chat_history = st.session_state.chats.get(st.session_state.current_chat, [])
    agent.init_chat_history(chat_history)

### INITIALIZATION END ###
if not st.session_state.current_chat:
    new_chat_name = ''
    flag = True
    while flag or new_chat_name in st.session_state.chats:
        new_chat_name = generate_chat_name()
        flag = False
        
    st.session_state.chats[new_chat_name] = []  # Initialize new chat history
    st.session_state.current_chat = new_chat_name  # Switch to the new chat


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


# Delete Current Chat
if st.sidebar.button("Delete Current Chat", type="primary"):
    if st.session_state.current_chat:
        del st.session_state.chats[st.session_state.current_chat]
        st.session_state.current_chat = None
        save_chats(CHAT_HISTORY_FILE)
        st.rerun()

# Add a divider between sections
st.sidebar.markdown("---")

# List of Existing Chats
st.sidebar.subheader("Your Chats")
for chat_name in st.session_state.chats.keys():
    if "Chat #" not in chat_name:
        if st.sidebar.button(chat_name, key=f"chat_button_{chat_name}"):
            st.session_state.current_chat = chat_name  # Switch to the selected chat

### SIDEBAR END ###

### MAIN CHAT INTERFACE BEGIN ###

st.title("Cal-AI Chat")

if st.session_state.current_chat:
    # microphone button
    float_init()
    footer_container = st.container()

    # Инициализация old_bytes через st.session_state, если еще не инициализирован
    if "old_bytes" not in st.session_state:
        st.session_state.old_bytes = None

    with footer_container:
        col1, col2 = st.columns([4, 1])
        with col2:
            audio_bytes = audio_recorder(text="", icon_size="2x", key="recorder")

    footer_container.float(
        "position: fixed; bottom: 15%; left: 80%; transform: translateX(-50%); width: 50%; display: flex; align-items: center; justify-content: space-between;"
    )


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
        if len(st.session_state.chats[st.session_state.current_chat])==2:
            context = "[User: " + prompt + "\nAssistant" + ans + "]"
            new_name = generate_chat_name(context, llm)
            st.session_state.chats[new_name] = st.session_state.chats[st.session_state.current_chat]
            del st.session_state.chats[st.session_state.current_chat]
            st.session_state.current_chat = new_name
            if st.sidebar.button(new_name, key=f"chat_button_{new_name}"):
                pass

        save_chats(CHAT_HISTORY_FILE)
    
    # Проверка, изменились ли аудио данные
    if audio_bytes != st.session_state.old_bytes:
        # Save and process the audio file
        print(st.session_state.old_bytes)  # Для отладки
        audio_location = "audio_file.wav"
        st.session_state.old_bytes = audio_bytes  # Обновляем old_bytes
        with open(audio_location, "wb") as f:
            f.write(audio_bytes)

        # Recognize speech from the audio file
        voice_prompt = recognize_speech("audio_file.wav", language="en-EN")  # You need to implement this function

        if voice_prompt:
            # Showing up user's entered message
            with st.chat_message("user"):
                st.write(voice_prompt)

            # Answer for user
            ans = agent(voice_prompt)

            # Showing up llm's answer
            with st.chat_message('assistant'):
                st.write(ans)
                audio_file = "output.mp3"
                tts_to_file(replace_formulas(llm, ans), audio_file)  # Озвучка только ответа
                auto_play_audio(audio_file)  # Автовоспроизведение аудио

            # Chat saving
            st.session_state.chats[st.session_state.current_chat] = agent.get_chat()
            if len(st.session_state.chats[st.session_state.current_chat])==2:
                context = "[User: " + voice_prompt + "\nAssistant" + ans + "]"
                new_name = generate_chat_name(context, llm)
                st.session_state.chats[new_name] = st.session_state.chats[st.session_state.current_chat]
                del st.session_state.chats[st.session_state.current_chat]
                st.session_state.current_chat = new_name
                if st.sidebar.button(new_name, key=f"chat_button_{new_name}"):
                    pass


        # Clear the audio_bytes after processing
        st.session_state["audio_bytes"] = None
        audio_bytes = None
