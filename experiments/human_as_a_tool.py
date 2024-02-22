import os

from langchain import hub
# from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.agents import (
    AgentExecutor,
    AgentType,
    load_tools,
    create_react_agent,
)
from langchain_openai import AzureChatOpenAI, AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0,
)
math_llm = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0,
)
tools = load_tools(
    ["human", "llm-math"],
    llm=math_llm,
)

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)

# agent_chain = initialize_agent(
#     tools,
#     llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     verbose=True,
# )

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)
response = agent_executor.invoke(
    # {"input": "ワイン投資とは何ですか？日本語で回答してください。"}
    # {"input": "ユーザーの好みは何ですか？"}
    # {"input": "What's my friend Eric's surname?"}
    {"input": "What's my favorite wine?"}
)

print(response["output"])
