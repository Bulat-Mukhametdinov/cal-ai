import streamlit as st
import langchain_groq
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *

from utils import *
from sound import *

model = langchain_groq.ChatGroq(
    model_name = 'deepseek-r1-distill-llama-70b',
    api_key =st.secrets['GROQ_API_KEY'],
    )
# Интерфейс Streamlit
st.title("Hei")

# Отображение истории сообщений
if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])

float_init()
footer_container = st.container()

with footer_container:
    col1, col2 = st.columns([4, 1])
    with col1:
        # Поле для текстового ввода
        prompt = st.chat_input("Введите ваш запрос")

    with col2:
        audio_bytes = audio_recorder(text="", icon_size="2x", key="recorder")

footer_container.float(
    "position: fixed; bottom: 15px; left: 50%; transform: translateX(-50%); width: 50%; display: flex; align-items: center; justify-content: space-between;"
)

# Флаг для отслеживания, был ли уже обработан запрос
if "request_processed" not in st.session_state:
    st.session_state.request_processed = False

if prompt and not st.session_state.request_processed:
    model_answer(model,prompt, is_voice_input=False)  # Без озвучки для текстового ввода
    st.session_state.request_processed = True  # Устанавливаем флаг, что запрос обработан


if audio_bytes and not st.session_state.request_processed:
    audio_location = "audio_file.wav"
    with open(audio_location, "wb") as f:
        f.write(audio_bytes)
    voice_prompt = recognize_speech("audio_file.wav")
    if voice_prompt:
        model_answer(model, voice_prompt, is_voice_input=True)  # Озвучка для голосового ввода
        st.session_state.request_processed = True  # Устанавливаем флаг, что запрос обработан


# Сброс флага после обработки запроса
if st.session_state.request_processed:
    st.session_state.request_processed = False