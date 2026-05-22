import os
import logging
import hashlib
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from services.rag_service import add_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def ingest_docs(folder="docs/"):
    docs = []

    # TXT files
    txt_loader = DirectoryLoader(folder, glob="**/*.txt", loader_cls=TextLoader)
    txt_docs = txt_loader.load()
    docs.extend(txt_docs)
    logger.info(f"Loaded {len(txt_docs)} TXT files")    

    # PDF files
    pdf_loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()
    docs.extend(pdf_docs)
    logger.info(f"Loaded {len(pdf_docs)} PDF files")
    

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [hashlib.md5(c.page_content.encode()).hexdigest() for c in chunks]

    add_documents(texts, ids, metadatas)

    logger.info(f"✅ Ingested {len(chunks)} chunks")