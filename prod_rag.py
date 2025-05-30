import redis
import streamlit as st
import os
import nltk
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.cache import RedisCache
from langchain.globals import set_llm_cache

# Set up Redis with error handling
try:
    redis_client = redis.Redis(host="localhost", port=6379, db=0, socket_connect_timeout=5, socket_timeout=5)
    redis_client.ping()
except redis.exceptions.TimeoutError as e:
    st.error("Redis connection timed out. Please check your Redis server.")
    st.stop()
except redis.exceptions.ConnectionError as e:
    st.error("Redis server is unreachable. Please ensure Redis is running and accessible.")
    st.stop()

# Download NLTK tokenizer
nltk.download('punkt')

# Get Google API Key
GOOGLE_API_KEY = st.text_input("Enter your password", type="password")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Template for prompt
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to
answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum
and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

def load_excel(file_path):
    loader = UnstructuredExcelLoader(file_path)
    return loader.load()

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def build_semantic_retriever(documents):
    if documents is None:
        raise ValueError("Expected list of documents, got None.")
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore.from_documents(documents, embeddings)
    return vector_store.as_retriever()

def build_bm25_retriever(documents):
    return BM25Retriever.from_documents(documents, preprocess_func=word_tokenize)

def build_hybrid_retriever(documents):
    semantic_retriever = build_semantic_retriever(documents)
    bm25_retriever = build_bm25_retriever(documents)
    return EnsembleRetriever(retrievers=[semantic_retriever, bm25_retriever], weights=[0.3, 0.7])

def get_or_cache_answer(question, documents):
    cache_key = f"answer:{question}"
    try:
        cached_answer = redis_client.get(cache_key)
        if cached_answer:
            return cached_answer.decode("utf-8")

        retriever = build_hybrid_retriever(documents)
        context = retriever.get_relevant_documents(question)
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

        response = llm.invoke(prompt.format(question=question, context=context))
        redis_client.set(cache_key, response.content)

        return response.content
    except redis.exceptions.TimeoutError:
        return "Error: Redis connection timed out."
    except redis.exceptions.ConnectionError:
        return "Error: Cannot connect to Redis."

# Streamlit UI
st.title("Excel QA APP")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    file_path = os.path.join("./", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "retriever" not in st.session_state:
        documents = load_excel(file_path)
        chunked_documents = split_text(documents)
        retriever = build_hybrid_retriever(chunked_documents)

        st.session_state.documents = chunked_documents
        st.session_state.retriever = retriever

question = st.chat_input("Ask a question based on uploaded excel file")

if question and "retriever" in st.session_state:
    st.chat_message("user").write(question)
    related_docs = st.session_state.retriever.invoke(question)
    answer = get_or_cache_answer(question, related_docs)
    st.chat_message("assistant").write(answer)
