import os
from dotenv import load_dotenv

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

# from langchain.chat_models import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage

# https://learn.microsoft.com/ja-jp/azure/ai-services/openai/reference
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"

SAMPLE_HISTORY = [
    ({"input": "こんにちは"}, {"output": "こんにちは！元気ですか？"}),
    ({"input": "私の名前は花子です"}, {"output": "よろしくね！"}),
]

load_dotenv()

llm = ChatOpenAI(
    temperature=0,
    # deployment_name=os.environ["GPT_35_MODEL_NAME"],
)

# https://python.langchain.com/docs/modules/memory/adding_memory
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="あなたの名前は太郎です。"
            # content="You are a chatbot having a conversation with a human."
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{input}"
            # "{human_input}"
        ),  # Where the human input will injected
    ]
)

# TODO: 要約指示文が英語のため、英語で要約されてしまう。カスタムプロンプトで日本語で要約するように指示は可能。
# 但し、会話履歴を UI に表示するためには、会話履歴と要約した結果
memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=10, memory_key="chat_history", return_messages=True
)

# for human_input, ai_output in SAMPLE_HISTORY:
#     memory.save_context(human_input, ai_output)

chain = ConversationChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

while True:
    command = input("You: ")
    if command == "exit":
        break
    # result = chain.predict(human_input=command)
    result = chain.predict(input=command)
    print(f"AI: {result}")
