import os
import sys
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoTokenizer
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import gradio
from typing import Callable, Dict, List, Optional, Union
from langchain.vectorstores import Chroma
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.llms import llamacpp
from langchain_community.document_loaders import DirectoryLoader
from utills import get_session_history, load_documents, split_docs, retriever_from_chroma, history_aware_retriever,chroma_db




script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data")
model_path = os.path.join(script_dir, '/mistral-7b-v0.1-layla-v4-Q4_K_M.gguf.2')
store = {}

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print(script_dir)
print(data_path)


documents = []
    
for filename in os.listdir(data_path):

    if filename.endswith('.txt'):

        file_path = os.path.join(data_path, filename)

        documents = TextLoader(file_path).load()

        documents.extend(documents)
docs = split_docs(documents, 450, 20)
chroma_db = chroma_db(docs,hf)
retriever = retriever_from_chroma(chroma_db, "mmr", 6)


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = llamacpp.LlamaCpp(
    model_path= model_path,
    n_gpu_layers=0,
    temperature=0.1,
    top_p=0.5,
    n_ctx=31000,
    max_tokens=250,
    repeat_penalty=1.7,
    stop=["", "Instruction:", "### Instruction:", "###<user>", "</user>"],
    callback_manager=callback_manager,
    verbose=False,
)


contextualize_q_system_prompt = """Given a context, chat history and the latest user question
which maybe reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is."""

ha_retriever = history_aware_retriever(llm, retriever, contextualize_q_system_prompt)

qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be as informative as possible, be polite and formal.\n{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(ha_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

conversational_rag_chain.invoke(
    {"input": "At what time scale a pre trail investigation has to be made ?"},
    
    config={
        "configurable": {"session_id": "99"}
    }, 
)["answer"]