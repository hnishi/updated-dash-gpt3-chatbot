# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
# os.environ['FAISS_NO_AVX2'] = '1'
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings


# os.environ["AZURE_OPENAI_API_KEY"] = "..."
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://<your-endpoint>.openai.azure.com/"
file_path = "./data/Jun_Tachibana.pdf"

loader = UnstructuredFileLoader(file_path)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
# embeddings = OpenAIEmbeddings()

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2024-02-15-preview",
)

db = FAISS.from_documents(docs, embeddings)
db.save_local("./data/faiss_index")
