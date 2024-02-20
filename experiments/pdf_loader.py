import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

file_path = "./data/Jun_Tachibana.pdf"
# file_path = "./data/review_202202_vol1_06.pdf"

llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0,
)

loader = UnstructuredFileLoader(file_path)
docs = loader.load()

# print(f"number of docs: {len(docs)}")
# # print("--------------------------------------------------")
# # print(docs[0].page_content)
# print("number of characters", len(docs[0].page_content))

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splitted_docs = text_splitter.split_documents(docs)

# print("--------------------------------------------------")
# print(splitted_docs[0].page_content)
# print(f"number of docs: {len(splitted_docs)}")
# print("number of characters", len(splitted_docs[0].page_content))

prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーから与えられる入力からプレゼン資料のテーマの候補をマークダウンの箇条書きで列挙してください。"),
    ("user", "{input}")
])

chain = prompt | llm 

# result = llm.invoke(f"")
result = chain.invoke({"input": splitted_docs[0].page_content})

print(result.content)