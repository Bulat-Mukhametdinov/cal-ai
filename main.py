import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
import langchain_groq

load_dotenv()
model = langchain_groq.ChatGroq(
    model_name = 'deepseek-r1-distill-llama-70b',
    api_key =os.getenv('GROQ_API_KEY'),
    )


def response_generator(prompt): #генерация ответов
    messages = [
        HumanMessage(prompt),
        SystemMessage("""You are an expert in calculus. User will ask you questions and you'll need to answer them, using your knowledge in math and the
                      context, if it exists. There are some rules you MUST follow in your response:
                      -Write ALL of your formulas on the correct latex, so streamlit.write() will show them correctly. Every latex-expression need to be framed with $.
                      -If there is a formula in your answer, replace all the [] and () with $   .
                      """),
    ]

    response = model.invoke(messages).content
    return response


def preprocess_think_tags(text): #обработка текста, чтобы были разные цвета у размышлений и ответа
    # Заменяем <think>...</think> на HTML с CSS-стилизацией
    if '</think>' in text:
        processed_text = text.replace("<think>", '<div style="color:red;">Размышления: </div><div style="font-size: 0.8em; opacity: 0.5;">')
        processed_text = processed_text.replace("</think>", '</div><div style="color:red;">Ответ: </div>\n<div style="font-size: 1em;">')
        processed_text += '\n</div>'
    else:
        processed_text = text.replace("<think>", '')
        processed_text = '<span style="color: yellow; font-style: Roboto;">' + processed_text
        processed_text += '</span>'
    return processed_text

#def llm_latex_prepocessing():#второй запрос, который вынет все формулы и заменит их на корректные.
def render_text_with_latex(text):
    for symb in ['\[', '\]', '\(', '\)']:
        text = text.replace(symb, '$')
    return text

     
def model_answer(prompt):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])

    st.session_state.messages.append({'role':'user', 'content': prompt})
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.chat_message('assistant'):
        messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
        ]
        prompt = ''
        for i in range(len(messages)):
            prompt += str(messages[i])
        ans = response_generator(prompt)

        ans_with_html =  render_text_with_latex(preprocess_think_tags(ans))
        
        st.write(ans_with_html, unsafe_allow_html=True)
        print(ans)
        ans = ans.split('</think>')[1]
    st.session_state.messages.append({'role':'assistant', 'content':ans})
    return ans

def reset_conversation():
  st.session_state.conversation = None
  st.session_state.messages = None
st.button('Reset Chat', on_click=reset_conversation)


st.title("hei")

if prompt := st.chat_input("What is up"):
    ans = model_answer(prompt)





