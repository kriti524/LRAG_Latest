from langchain_community.vectorstores import Chroma
from core.config import CHROMA_DIR
from core.embeddings import get_embeddings

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings()
    )

def add_documents(texts, ids, metadatas=None):
    vectorstore = get_vectorstore()
    vectorstore.add_texts(
        texts=texts,
        ids=ids,
        metadatas=metadatas
    )

def retrieve(query, k=3):
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search(query, k=k)