import os
import sys
from langchain.text_splitter import TokenTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Callable, Dict, List, Optional, Union
from langchain.vectorstores import Chroma
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.llms import llamacpp


store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]




def load_documents(data_path):
    try:
        document_loader = PyPDFDirectoryLoader(data_path)
        return document_loader.load()
    except Exception as e:
        print(f"Error loading documents from {data_path}: {e}")
        return None  # or handle the error in an appropriate manner    



def split_docs(documents, chunk_size, chunk_overlap):
    try:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n \n \n", "\n \n", "\n1", "(?<=\. )", " ", ""]
        )
        docs = text_splitter.split_documents(documents)
        return docs
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return []  # or handle the error in an appropriate manner


def chroma_db(docs, embeddings):
    try:
        vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory="docs/chroma/"
        )
        return vectordb
    except Exception as e:
        print(f"Error creating Chroma vector database: {e}")
        return None  # or handle the error in an appropriate manner


def retriever_from_chroma(vectordb, search_type, k):
    retriever = vectordb.as_retriever(search_type=search_type, search_kwargs={"k": k})
    return retriever    


def history_aware_retriever(llm, retriever, contextualize_q_system_prompt):
    try:
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        return history_aware_retriever
    except Exception as e:
        print(f"Error creating history-aware retriever: {e}")
        return None  # or handle the error in an appropriate manner




def echo(question, history):
    ai_message = rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), ai_message["answer"]])
    return ai_message['answer']
