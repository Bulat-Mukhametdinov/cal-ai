import streamlit as st
import re
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
import langchain_groq
import speech_recognition as sr
import pyttsx3
from audio_recorder_streamlit import audio_recorder
from langchain import PromptTemplate
import base64

# Инициализация модели
model = langchain_groq.ChatGroq(
    model_name='deepseek-r1-distill-llama-70b',
    api_key=st.secrets['GROQ_API_KEY'],
)

def response_generator(prompt):  # Генерация ответов
    messages = [
        HumanMessage(prompt),
        SystemMessage("""You are an expert in calculus. User will ask you questions and you'll need to answer them, using your knowledge in math and the
                      context, if it exists. There are some rules you MUST follow in your response:
                      -Write ALL of your formulas on the correct latex, so .streamlit.write() will show them correctly. Every latex-expression need to be framed with $.
                      -If there is a formula in your answer, replace all the [] and () with $$.
                      """),
    ]

    response = model.invoke(messages).content
    return response

def preprocess_think_tags(text):  # Обработка текста, чтобы были разные цвета у размышлений и ответа
    if '</think>' in text:
        # Разделяем размышления и ответ
        think_part, answer_part = text.split('</think>', 1)
        processed_text = (
            f'<div style="color:red;">Размышления: </div><div style="font-size: 0.8em; opacity: 0.5;">{think_part}</div>'
            f'<div style="color:red;">Ответ: </div>\n<div style="font-size: 1em;">{answer_part}</div>'
        )
        return processed_text, answer_part.strip()  # Возвращаем обработанный текст и только ответ
    else:
        processed_text = '<span style="color: yellow; font-style: Roboto;">' + text + '</span>'
        return processed_text, text.strip()  # Возвращаем текст без размышлений

def recognize_speech(wav_file_path, language="ru-RU"):
    """
    Расшифровывает WAV-файл в текст.

    :param wav_file_path: Путь к WAV-файлу.
    :param language: Язык распознавания (по умолчанию "ru-RU").
    :return: Расшифрованный текст или None, если произошла ошибка.
    """
    recognizer = sr.Recognizer()

    try:
        # Открываем WAV-файл
        with sr.AudioFile(wav_file_path) as source:
            audio = recognizer.record(source)

        # Используем Google Speech Recognition для распознавания речи
        text = recognizer.recognize_google(audio, language=language)
        return text

    except sr.UnknownValueError:
        print("Не удалось распознать речь.")
        return None
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания речи: {e}")
        return None

def tts_to_file(text, file_path="output.mp3"):  # Преобразование текста в аудио и сохранение в файл
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 150)
    engine.save_to_file(text, file_path)
    engine.runAndWait()

def model_answer(prompt, is_voice_input=False):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Добавляем запрос пользователя в историю
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        prompt = ''
        for i in range(len(messages)):
            prompt += str(messages[i])
        ans = response_generator(prompt)

        # Обработка текста для отображения и извлечения ответа
        processed_text, answer_only = preprocess_think_tags(ans)

        # Отображение обработанного текста
        st.write(processed_text, unsafe_allow_html=True)

        # Сохраняем ответ для последующего использования
        st.session_state.messages.append({'role': 'assistant', 'content': ans})

        # Если запрос был через микрофон, озвучиваем ответ
        if is_voice_input:
            audio_file = "output.mp3"
            tts_to_file(replace_formulas(model, answer_only), audio_file)  # Озвучка только ответа
            auto_play_audio(audio_file)  # Автовоспроизведение аудио

    return ans


# убираем все формулы(для программы Димона)
def replace_formulas(model, text):
    prompt_templ = """Ты — помощник по переводу математических формул в текст, который можно легко зачитать или озвучить. Твоя задача — преобразовать любое математическое выражение в простой текст, используя только буквы, цифры, пробелы и знаки препинания. Вот правила, которые нужно соблюдать:

    1. Убери все специальные символы :
        Замени символы $, ^, \sum, \infty, \frac и другие на слова или фразы.
        Например, "\sum" замени на "сумма", "^n" замени на "в степени n", "\infty" замени на "до бесконечности".
    2. Опиши формулу словами :
        Объясни каждый элемент формулы так, будто ты рассказываешь это человеку.
        Например, "f(x)" можно описать как "функция f от x".
    3.Избегай сложных конструкций :
        Используй только простые предложения, разделённые точками, запятыми или союзами.
    Пример стиля :
    Вход: f(x) = \sum_n=0^\infty \\fracf^(n)(a)n!(x - a)^n
    Выход: Функция f от x равна сумме бесконечного ряда Каждый член ряда равен n-й производной функции f в точке a делённой на факториал числа n и умноженной на выражение x минус a в степени n Сумма берётся по всем значениям n начиная с нуля
    Пример для практики :
    Вход: e^x = \sum_n=0^\infty \\fracx^nn!
    Выход: Экспонента от x равна сумме бесконечного ряда Каждый член ряда равен x в степени n делённому на факториал числа n Сумма берётся по всем значениям n начиная с нуля
    Твоя задача :
    Получить математическую формулу в виде текста.
    Преобразовать её в простой текст, который можно легко зачитать или озвучить.
    Убедиться, что в тексте нет специальных символов, только буквы, цифры, пробелы и знаки препинания. 

    Выведи полностью следующе сообщение с заменнеными формулами и ничего больше, НИЧЕГО:
    {message}"""
    prompt = PromptTemplate(input_variables=['message'], template=prompt_templ)
    messages = prompt.format(
        message=text
    )
    response = model.invoke(messages).content
    return response.split('</think>')[-1]

def auto_play_audio(audio_file):
    with open(audio_file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    audio_html = f'<audio src="data:audio/mp3;base64,{base64_audio}" controls autoplay>'
    st.markdown(audio_html, unsafe_allow_html=True)

# Сброс диалога
def reset_conversation():
    st.session_state.conversation = None
    st.session_state.messages = []

# Интерфейс Streamlit
st.title("Hei")

# Отображение истории сообщений
if "messages" in st.session_state:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])

# Создание колонок для текстового ввода и кнопки записи
col1, col2 = st.columns([4, 1])  # Пропорция 4:1 для текстового ввода и кнопки записи

# Поле для текстового ввода

if prompt := st.chat_input("Введите ваш запрос"):
    model_answer(prompt, is_voice_input=False)  # Без озвучки для текстового ввода

with col2:
    # Кнопка записи
    audio_bytes = audio_recorder(text="", icon_size="2x", key="recorder")
    if audio_bytes:
        audio_location = "audio_file.wav"
        with open(audio_location, "wb") as f:
            f.write(audio_bytes)
        voice_prompt = recognize_speech("audio_file.wav")
        if voice_prompt:
            with col1:
                model_answer(voice_prompt, is_voice_input=True)  # Озвучка для голосового ввода

# Кнопка для сброса чата
st.button('Reset Chat', on_click=reset_conversation)
