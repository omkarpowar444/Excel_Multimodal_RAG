import redis.client
import streamlit as st
import os
import nltk
import redis
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


redis_client=redis.Redis(host="gifted_cray" , port=6379 , db=0)

nltk.download('punkt')

GOOGLE_API_KEY=st.text_input("Enter your password" , type="password")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

template="""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to
answer the question. If you dont know the answer, just say that you dont know. Use three sentences maximum
and keep the answer concise.
Question:{question}
Context:{context}
Answer:
"""

def load_excel(file_path):
    loader=UnstructuredExcelLoader(file_path)
    return loader.load()

def split_text(documents):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,
                                                 chunk_overlap=50,
                                                 add_start_index=True)
    
    return text_splitter.split_documents(documents)
    
def build_semantic_retriever(documents):
    if documents is None:
        raise ValueError("Expected list of documents, got None.")
    embeddings=HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store=InMemoryVectorStore.from_documents(documents,embeddings)
    return vector_store.as_retriever()

def build_bm25_retriever(documents):
    return BM25Retriever.from_documents(documents,preprocess_func=word_tokenize)

def build_hybrid_retriever(documents):
    semantic_retriever=build_semantic_retriever(documents)
    bm25_retriever=build_bm25_retriever(documents)
    return EnsembleRetriever(retrievers=[semantic_retriever,bm25_retriever], weights=[0.3 , 0.7])

def get_or_cache_answer(question,documents):
    cache_key=f"answer:{question}"
    cached_answer=redis_client.get(cache_key)

    if cached_answer:
        return cached_answer.decode("utf-8")
    #not in cache memory so will use llm now to give answer

    retriever=build_hybrid_retriever(documents)
    context=retriever.get_relevant_documents(question)
    prompt=ChatPromptTemplate.from_template(template)
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0)

    response=llm.invoke(prompt.format(question=question,context=context))

    redis_client.set(cache_key,response.content)

    return response.content


#   STREAMLIT UI

st.title("Excel QA APP")

uploaded_file=st.file_uploader("Upload an Excel file",type=["xlsx","xls"])


if uploaded_file:
    file_path=os.path.join("./",uploaded_file.name)

    with open(file_path,"wb") as f:
        f.write(uploaded_file.getbuffer())

    if "retriever" not in st.session_state:
        documents=load_excel(file_path)
        chunked_documents=split_text(documents)
        retriever=build_hybrid_retriever(chunked_documents)

        st.session_state.documents=chunked_documents
        st.session_state.retriever=retriever

question=st.chat_input("Ask a question based on uploaded excel file")

if question and "retriever" in st.session_state:
    st.chat_message("user").write(question)

    related_docs=st.session_state.retriever.invoke(question)

    answer=get_or_cache_answer(question,related_docs)

    st.chat_message("assistant").write(answer)


    

