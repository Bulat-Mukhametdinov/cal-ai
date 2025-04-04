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

def tts_to_file(text, file_path="output.mp3"):  # Преобразование текста в аудио и сохранение в файл
    # Инициализация движка TTS
    engine = pyttsx3.init()
    language = 'ru'

    # Получение списка доступных голосов
    voices = engine.getProperty('voices')

    # Поиск английского голоса
    selected_voice = None
    for voice in voices:
        if language in voice.id.lower():  # Проверка, содержит ли ID голоса 'en' (английский)
            selected_voice = voice.id
            break

    if selected_voice is None:
        raise ValueError("Английский голос не найден в системе. Убедитесь, что он установлен.")

    # Установка выбранного голоса и скорости речи
    engine.setProperty('voice', selected_voice)
    engine.setProperty('rate', 150)  # Скорость речи

    # Сохранение аудио в файл
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
        text = recognizer.recognize_google(audio, language=language)
        return text

    except sr.UnknownValueError:
        print("Не удалось распознать речь.")
        return None
    except sr.RequestError as e:
        print(f"Ошибка сервиса распознавания речи: {e}")
        return None