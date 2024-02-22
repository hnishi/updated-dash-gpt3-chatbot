# Uncomment the following line if you need to initialize FAISS with no AVX2 optimization
# os.environ['FAISS_NO_AVX2'] = '1'

from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings


embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    openai_api_version="2024-02-15-preview",
)

db = FAISS.load_local("./data/faiss_index", embeddings)

query = "ワイン投資とは何ですか？"

docs = db.similarity_search(query)

print(docs[0].page_content)
