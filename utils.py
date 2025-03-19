import streamlit as st
import os
from langchain import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


#делаем из промпта юзера запрос для векторной базы
def question_generator(model, text):
    prompt_templ = """Ты — помощник по переформулировке вопросов и решений, связанных с математическим анализом, для работы с учебником. Твоя задача — преобразовать ввод пользователя в строгую, формальную формулировку, которая максимально соответствует стилю учебника по математическому анализу. Переформулировка должна быть оптимизирована для RAG-системы, работающей с этим учебником.

При переформулировке следуй этим правилам:

1. Используй строгий математический язык :
Все формулировки должны быть формальными, без разговорных оборотов.
Используй стандартные термины и обозначения из учебника (например, "предел последовательности", "непрерывность функции", "теорема Коши").
2. Устраняй двусмысленность :
Если ввод пользователя содержит неясные или неточные формулировки, переформулируй их так, чтобы они стали однозначными.
Убедись, что все переменные, функции и условия четко указаны.
3. Структурируй запрос :
Разделяй сложные вопросы или решения на логические части.
Если пользователь предоставляет доказательство или решение, выдели основные этапы рассуждений в виде отдельных шагов.
4. Определи цель запроса :
Если пользователь запрашивает проверку решения или доказательства, переформулируй запрос как проверку корректности выполненных шагов.
Если пользователь задает вопрос, переформулируй его как задачу или утверждение, требующее объяснения.
5. Сохраняй семантику :
Переформулировка должна точно передавать исходный смысл вопроса или решения, не добавляя новых интерпретаций.
6. Формат выхода :
Вывод должен быть кратким, формальным и готовым для векторизации.
Используй стиль учебника: определения, теоремы, задачи и примеры должны быть представлены в строгой форме.

Примеры работы:

Вход:
"Я пытался доказать, что если функция непрерывна на отрезке [a, b], то она достигает своего максимума и минимума. Вот мои шаги: сначала я взял произвольную точку c ∈ [a, b], потом показал, что значения f(x) ограничены, и нашел точку, где значение максимальное. Правильно ли я сделал?"
Выход:
"Проверить корректность доказательства теоремы Вейерштрасса о достижении непрерывной функцией своих экстремумов на отрезке [a, b]:
Функция f(x) непрерывна на отрезке [a, b].
Доказывается ограниченность значений f(x) на [a, b].
Доказывается существование точки c ∈ [a, b], где f(c) = max f(x)."

Вход:
"Как найти предел lim (x → 0) (1 - cos(x)) / x^2?"
Выход:
"Вычислить предел lim (x → 0) ((1 - cos(x)) / x^2)."

Вход:
"Я решаю задачу на нахождение площади под графиком y = ln(x) на отрезке [1, e]. Сначала я записал интеграл ∫[1, e] ln(x) dx, потом применил интегрирование по частям, но застрял. Как дальше?"
Выход:
"Решить задачу нахождения площади под графиком функции y = ln(x) на отрезке [1, e]:
Задан интеграл ∫[1, e] ln(x) dx.
Применено интегрирование по частям.
Требуется завершение вычисления интеграла."

Вход:
"Что такое производная функции f(x) в точке x = a?"
Выход:
"Определение производной функции f(x) в точке x = a:
Производная функции f(x) в точке x = a определяется как lim (h → 0) (f(a + h) - f(a)) / h, если этот предел существует."

Вот мой запрос, который необходимо переформулировать:
{query}
"""
    prompt = PromptTemplate(input_variables=['query'], template = prompt_templ)
    messages = prompt.format(
        query=text
    )
    response = model.invoke(messages).content
    return response.split('</think>')[-1]

#генерация ответов
def response_generator(model,prompt): 
    messages = [
        SystemMessage("""You are an expert in calculus. User will ask you questions and you'll need to answer them, using your knowledge in math and the
                      context, if it exists. There are some rules you MUST follow in your response:
                      -Write ALL of your formulas on the correct latex, so streamlit.write() will show them correctly. Every latex-expression need to be framed with $.
                      -If there is a formula in your answer, replace all the [] and () with $   .
                      """),
        HumanMessage(prompt),
    ]

    response = model.invoke(messages).content
    return response

#убираем все формулы(для программы Димона)
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
    prompt = PromptTemplate(input_variables=['message'], template = prompt_templ)
    messages = prompt.format(
        message=text
    )
    response = model.invoke(messages).content
    return response.split('</think>')[-1]


#обработка текста, чтобы были разные цвета у размышлений и ответа
def preprocess_think_tags(text): 
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

def render_text_with_latex(text):
    for symb in ['\[', '\]', '\(', '\)']:
        text = text.replace(symb, '$')
    return text

def model_answer(model, prompt):
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message['content'])

    
    with st.chat_message("user"):
        st.write(prompt)
    query = question_generator(model, prompt)
    print(query)
    st.session_state.messages.append({'role':'user', 'content': query})

    with st.chat_message('assistant'):
        messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
        ]
        prompt = ''
        for i in range(len(messages)):
            prompt += str(messages[i])
        ans = response_generator(model, prompt)
        ans_with_html =  render_text_with_latex(preprocess_think_tags(ans))
        st.write(ans_with_html, unsafe_allow_html=True)
        ans = ans.split('</think>')[-1]
        message_to_voice = replace_formulas(model, ans)
        ans_for_history = render_text_with_latex(ans).split('</think>')[-1]
    st.session_state.messages.append({'role':'assistant', 'content':ans_for_history})
    return (ans, message_to_voice)

# def reset_conversation():
#   st.session_state.conversation = None
#   st.session_state.messages = None

