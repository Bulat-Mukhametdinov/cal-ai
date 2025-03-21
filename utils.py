import streamlit as st
import random
import os
import json
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Save chat history to file
def save_chats(chats_save_dest):
    chat_history = st.session_state.chats
    chat_history_json = json.dumps({chat_name: [chat.model_dump() for chat in chat_history[chat_name]] for chat_name in chat_history}, indent=4)
    # Save to a file
    with open(chats_save_dest, "w") as f:
        f.write(chat_history_json)

# Load chat history from file on startup
def load_chats(chats_load_src):
    try:
        with open(chats_load_src, "r") as f:
            chats_data = json.load(f)
    except:
        return {}

    # Convert dictionaries back to message objects
    message_type_mapping = {
        "human": HumanMessage,
        "ai": AIMessage,
        "system": SystemMessage
    }
    chats = {}
    for chat_name in chats_data:
        chats[chat_name] = [message_type_mapping[msg["type"]](content=msg["content"]) for msg in chats_data[chat_name]]

    return chats


def generate_chat_name():
    name = f"Chat #{str(os.urandom(6).hex())}"
    return name


# #убираем все формулы(для программы Димона)
# def replace_formulas(model, text):
#     prompt_templ = """Ты — помощник по переводу математических формул в текст, который можно легко зачитать или озвучить. Твоя задача — преобразовать любое математическое выражение в простой текст, используя только буквы, цифры, пробелы и знаки препинания. Вот правила, которые нужно соблюдать:

#     1. Убери все специальные символы :
#         Замени символы $, ^, \sum, \infty, \frac и другие на слова или фразы.
#         Например, "\sum" замени на "сумма", "^n" замени на "в степени n", "\infty" замени на "до бесконечности".
#     2. Опиши формулу словами :
#         Объясни каждый элемент формулы так, будто ты рассказываешь это человеку.
#         Например, "f(x)" можно описать как "функция f от x".
#     3.Избегай сложных конструкций :
#         Используй только простые предложения, разделённые точками, запятыми или союзами.
#     Пример стиля :
#     Вход: f(x) = \sum_n=0^\infty \\fracf^(n)(a)n!(x - a)^n
#     Выход: Функция f от x равна сумме бесконечного ряда Каждый член ряда равен n-й производной функции f в точке a делённой на факториал числа n и умноженной на выражение x минус a в степени n Сумма берётся по всем значениям n начиная с нуля
#     Пример для практики :
#     Вход: e^x = \sum_n=0^\infty \\fracx^nn!
#     Выход: Экспонента от x равна сумме бесконечного ряда Каждый член ряда равен x в степени n делённому на факториал числа n Сумма берётся по всем значениям n начиная с нуля
#     Твоя задача :
#     Получить математическую формулу в виде текста.
#     Преобразовать её в простой текст, который можно легко зачитать или озвучить.
#     Убедиться, что в тексте нет специальных символов, только буквы, цифры, пробелы и знаки препинания. 
                      
#     Выведи полностью следующе сообщение с заменнеными формулами и ничего больше, НИЧЕГО:
#     {message}"""
#     prompt = PromptTemplate(input_variables=['message'], template = prompt_templ)
#     messages = prompt.format(
#         message=text
#     )
#     response = model.invoke(messages).content
#     return response.split('</think>')[-1]


# #обработка текста, чтобы были разные цвета у размышлений и ответа
# def preprocess_think_tags(text): 
#     # Заменяем <think>...</think> на HTML с CSS-стилизацией
#     if '</think>' in text:
#         processed_text = text.replace("<think>", '<div style="color:red;">Размышления: </div><div style="font-size: 0.8em; opacity: 0.5;">')
#         processed_text = processed_text.replace("</think>", '</div><div style="color:red;">Ответ: </div>\n<div style="font-size: 1em;">')
#         processed_text += '\n</div>'
#     else:
#         processed_text = text.replace("<think>", '')
#         processed_text = '<span style="color: yellow; font-style: Roboto;">' + processed_text
#         processed_text += '</span>'
#     return processed_text

# #заменяем скобки для латеха
# def render_text_with_latex(text):
#     for symb in ['\[', '\]', '\(', '\)']:
#         text = text.replace(symb, '$')
#     return text

#делаем пословный вывод
def write_with_delay(text):
    
    for word in text.split():
        yield st.write(word, unsafe_allow_html=True)
        time.sleep(0.05)
    return text

# #фулл ответ модельки
# def model_answer(model, prompt):
    
#     current_chat_history = st.session_state.chats[st.session_state.current_chat]
#     with st.chat_message("user"):
#         st.write(prompt)
#     query = question_generator(model, prompt)
#     print(query)
#     current_chat_history.append({'role':'user', 'content': prompt})

#     with st.chat_message('assistant'):
#         messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in current_chat_history
#         ]
#         prompt = ''
#         for i in range(len(messages)):
#             prompt += str(messages[i])
#         ans = response_generator(model, prompt)
#         ans_with_html =  render_text_with_latex(preprocess_think_tags(ans))
#         st.write(ans_with_html, unsafe_allow_html=True)
#         ans = ans.split('</think>')[-1]
#         message_to_voice = replace_formulas(model, ans)
#         ans_for_history = render_text_with_latex(ans).split('</think>')[-1]
        
#     current_chat_history.append({'role':'assistant', 'content':ans_for_history})
#     return (ans, message_to_voice)