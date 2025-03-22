import streamlit as st
import speech_recognition as sr
import pyttsx3
import base64


def auto_play_audio(audio_file):
    with open(audio_file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    audio_html = f'<audio src="data:audio/mp3;base64,{base64_audio}" controls autoplay>'
    st.markdown(audio_html, unsafe_allow_html=True)

def tts_to_file(text, file_path="output.mp3"):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    # Выбор русского голоса
    for voice in voices:
        if "russian" in voice.name.lower():
            engine.setProperty('voice', voice.id)
            break

    engine.setProperty('rate', 150)  # Скорость речи
    engine.save_to_file(text, file_path)
    engine.runAndWait()

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
        text = recognizer.recognize_google(audio, language="ru-RU")
        return text

    except sr.UnknownValueError:
        st.write(f'<span style="color: red; opacity: 0.5;">Speech recognition failed</span>', unsafe_allow_html=True)
        return None
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания речи: {e}")
        return None