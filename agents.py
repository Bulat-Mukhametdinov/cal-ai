from tools import tools
from llm import llm

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class AgentAnswerPipeline():
    # Initialize translation prompt
    translation_prompt = PromptTemplate(template=
            "You are helpful assistant\n"
            "User input: {input}\n"
            "Response: {answer}\n"
            "Answer with translated (if required or original) response. Before your answer say 'My answer:'\n",
        input_variables=["input", "answer"]
    )

    def proccess_output(output):
        return output.split("My answer: ")[-1]

    # Create translation chain
    translate = translation_prompt | llm | StrOutputParser() | proccess_output

    # Load the ReAct Agent Prompt
    react_docstore_prompt = hub.pull("hwchase17/react-chat")

    # Create the chat ReAct Agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_docstore_prompt,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, handle_parsing_errors=True,
    )

    first_system_prompt = SystemMessage(content="You are helpful calculus assistant. Use latex format for all your formulas.")

    def __init__(self, debug=True):
        """set debug=True if you want to show agent's thoughts in console output"""

        self.agent_executor.set_verbose(debug)
        self.chat_history = [self.first_system_prompt]
    

    def __call__(self, query):
        response = self.agent_executor.invoke(
            {"input": query, "chat_history": self.chat_history})

        response = self.translate.invoke({"input": query, "answer": response["output"]})
        
        # Update history
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=response))
    
        return response

    def init_chat_history(self, history:list) -> None:
        self.chat_history = [self.first_system_prompt] + history
    
    def get_chat(self):
        return self.chat_history[1:]