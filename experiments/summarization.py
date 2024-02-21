# https://python.langchain.com/docs/use_cases/summarization
# https://zenn.dev/tsuzukia/articles/05bfdcfcf5bd68

import os
from textwrap import dedent

# from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader

# from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document


load_dotenv()

url = "https://hayatoito.github.io/2020/investing/"

# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0,
)


loader = WebBaseLoader(url)
raw_docs = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1500,
    chunk_overlap=100,
    separator=".",
)
texts = text_splitter.split_text(raw_docs[0].page_content)
docs = [Document(page_content=t) for t in texts]


def simple():
    chain = load_summarize_chain(
        llm, chain_type="map_reduce"
    )  # stuff, map_reduce, refine

    chain.invoke(docs)


def map_reduce():
    # map_prompt_template = """以下の文章をテーマ毎にまとめてく下さい。
    # map_prompt_template = """以下の文章の重要部分のみを具体的かつ簡潔まとめてく下さい。
    map_prompt_template = dedent(
        """
        以下の文章を情報を落とさないまま、プレゼンテーションの原稿にして下さい。
        ------
        {text}
        ------
        """
    )[1:-1]

    # map_combine_template = """以下の文章をテーマ毎にまとめてください。
    map_combine_template = dedent(
        """
        以下の文章からプレゼンテーションの原稿の最終版を作成して下さい。
        ------
        {text}
        ------
        """
    )[1:-1]

    map_first_prompt = PromptTemplate(
        template=map_prompt_template, input_variables=["text"]
    )
    map_combine_prompt = PromptTemplate(
        template=map_combine_template, input_variables=["text"]
    )

    map_chain = load_summarize_chain(
        llm=llm,
        reduce_llm=llm,
        collapse_llm=llm,
        chain_type="map_reduce",
        map_prompt=map_first_prompt,
        combine_prompt=map_combine_prompt,
        collapse_prompt=map_combine_prompt,
        token_max=5000,
        verbose=True,
    )

    result = map_chain.invoke({"input_documents": docs}, return_only_outputs=True)
    print(result["output_text"])


def refine():
    refine_first_template = """以下の文章をテーマ毎にまとめて下さい。
    ------
    {text}
    ------
    """
    refine_template = """以下の文章をテーマ毎にまとめて下さい。
    ------
    {existing_answer}
    {text}
    ------
    """

    refine_first_prompt = PromptTemplate(
        input_variables=["text"], template=refine_first_template
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"], template=refine_template
    )

    refine_chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=refine_first_prompt,
        refine_prompt=refine_prompt,
        verbose=True,
    )

    result = refine_chain.invoke({"input_documents": docs}, return_only_outputs=True)
    print(result["output_text"])


# simple()
map_reduce()
# refine()
