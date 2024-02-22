import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    Tool,
    tool,
    create_react_agent,
)  # create_openai_functions_agent  # create_json_chat_agent

# from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.pydantic_v1 import BaseModel, Field


python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)


class TranslateInput(BaseModel):
    text: str = Field(description="any text")
    language: str = Field(description="language into which the text will be translated")


# @tool("translate-tool", args_schema=TranslateInput, return_direct=True)
def translate(text: str, language: str) -> str:
    """Translate a given text into a given language."""
    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", f"Please translate user's messages into {language}."),
            ("user", "{input}"),
        ]
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    response = chain.invoke({"input": text})
    return response


@tool
def translate2en(text: str) -> str:
    """Translate a text into English."""
    return translate(text, "English")


@tool
def translate2jp(text: str) -> str:
    """Translate a text into Japanese."""
    return translate(text, "Japanese")


# print(translate.name)
# print(translate.description)
# print(translate.args)
# print(translate("ワイン投資とは何ですか？", "English"))
print(translate2en.name)
print(translate2en.description)
print(translate2en.args)
# print(translate2en("ワイン投資とは何ですか？"))
# exit()

# translate_tool = Tool(
#     name="Translate tool",
#     func=lambda text, language=language: str(translate(text, language)),
#     description="Translate a given text into a given language.",
# )

tools = [TavilySearchResults(max_results=1), repl_tool, translate2en, translate2jp]

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0,
)

# prompt = hub.pull("hwchase17/react-chat-json")
# agent = create_json_chat_agent(llm, tools, prompt)
# prompt = hub.pull("hwchase17/openai-tools-agent")
# agent = create_openai_tools_agent(llm, tools, prompt)
# prompt = hub.pull("hwchase17/openai-functions-agent")
# agent = create_openai_functions_agent(llm, tools, prompt)
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)


# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)
response = agent_executor.invoke(
    # {"input": "ワイン投資とは何ですか？日本語で回答してください。"}
    {"input": "ワイン投資に関する4択問題を考えてください。"}
)
# response = agent_executor.invoke({"input": "1234*321 の答えは何ですか？日本語で答えてください。"})
# response = agent_executor.invoke({"input": "Answer 1234*321"})
print(response["intermediate_steps"])
print(response["output"])


# print(python_repl.run("print(1234*321)"))
