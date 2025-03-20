import json
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Sample chat history
chat_history = [SystemMessage(content="Welcome to the chat!"),
    HumanMessage(content="Hello! How are you?"),
    AIMessage(content="I'm good! How can I assist you today?")]

for i in list(range(1, 5)) + list(range(10, 15)):
    print(i)
