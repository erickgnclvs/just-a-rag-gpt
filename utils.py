import os
import requests
from typing import List
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI


DATA_DIR = "data/"


def load_documents(data_dir=DATA_DIR) -> List[str]:
    docs = []
    rag_file_url = os.environ.get("RAG_FILE_URL")
    temp_file_path = os.path.join(data_dir, "temp_rag_download.md")

    if not rag_file_url:
        print("Error: RAG_FILE_URL environment variable not set.")
        return docs

    print(f"Loading RAG document from URL: {rag_file_url}")
    try:
        response = requests.get(rag_file_url)
        response.raise_for_status()
        rag_content = response.text

        try:
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(rag_content)
        except IOError as e:
            print(f"Error writing temporary RAG file {temp_file_path}: {e}")
            return docs

        loader = UnstructuredMarkdownLoader(temp_file_path)
        docs = loader.load()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching RAG document from URL {rag_file_url}: {e}")

    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                print(f"Error deleting temporary file {temp_file_path}: {e}")

    if not docs:
         print("Warning: Failed to load any RAG documents from the URL.")

    return docs


def get_vectorstore(docs, api_key):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectordb = Chroma.from_documents(texts, embeddings, persist_directory=".chroma_db")
    return vectordb


def get_retriever(vectordb):
    return vectordb.as_retriever()


def get_qa_chain(api_key, retriever):
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-preview-04-17", google_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain
