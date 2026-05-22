import os
import logging
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

from sentence_transformers import CrossEncoder

from services.rag_service import get_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# ---------------- GLOBALS ----------------
_chain = None
_memory = None

# ---------------- RERANKER ----------------
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ---------------- MEMORY ----------------
def get_memory():

    global _memory

    if _memory is None:

        _memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    return _memory


# ---------------- RERANK FUNCTION ----------------
def rerank_documents(query, docs, top_n=3):

    if not docs:
        return []

    pairs = [
        [query, doc.page_content]
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))

    scored_docs.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in scored_docs[:top_n]]


# ---------------- CHAIN ----------------
def get_chain():

    global _chain

    if _chain is None:

        logger.info("Initializing LLM chain")

        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            temperature=0.3,
            streaming=True
        )

        vectorstore = get_vectorstore()

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20
            }
        )

        prompt = PromptTemplate(
            input_variables=[
                "context",
                "question",
                "chat_history"
            ],
            template="""
You are a helpful assistant.

Use ONLY the provided context to answer.

If the answer is not present in the context,
say you don't know.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
"""
        )

        _chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=get_memory(),
            combine_docs_chain_kwargs={
                "prompt": prompt
            },
            return_source_documents=True
        )

    return _chain


# ---------------- GENERATE ANSWER ----------------
def generate_answer(question):

    chain = get_chain()

    logger.info(f"Question received: {question}")

    # Retrieve relevant documents
    docs = chain.retriever.invoke(question)
    if not docs:
        logger.warning("No relevant documents found")
        return "No relevant documents found.", []

    # Rerank retrieved docs
    reranked_docs = rerank_documents(
        question,
        docs,
        top_n=3
    )

    # Generate answer
    try:

        result = chain.combine_docs_chain.run(
            input_documents=reranked_docs,
            question=question,
            chat_history=get_memory().chat_memory.messages
        )
        logger.info("Answer generated successfully")

    except Exception as e:

        logger.error(f"LLM generation failed: {str(e)}")
        return "Error generating response.", []

    # Save conversation memory
    get_memory().chat_memory.add_user_message(
        question
    )

    get_memory().chat_memory.add_ai_message(
        result
    )

    # Extract unique sources
    sources = list(set([
        doc.metadata.get("source", "unknown")
        for doc in reranked_docs
    ]))

    return result, sources


# ---------------- CLEAR MEMORY ----------------
def clear_memory():

    global _chain, _memory

    _chain = None
    _memory = None


# ---------------- MODEL HEALTH CHECK ----------------
def check_model():

    try:

        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3")
        )

        llm.invoke("hi")
        logger.info("Model health check passed")

        return True

    except Exception as e:

        logger.error(f"Model check failed: {e}")

        return False