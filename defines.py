import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv
load_dotenv()
import requests
import faiss
import numpy
from langchain.llms import HuggingFaceHub
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings


def get_pdf_text(pdf_docs)->str:
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()

    return text



def get_text_chunked(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = '\n',
        chunk_size = 200,
        chunk_overlap = 20,
        length_function = len
    )
    chunked_text = text_splitter.split_text(raw_text) 

    return chunked_text



def get_vectorstore(chunked_text):
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    response = requests.post(api_url, headers=headers, json={"inputs": chunked_text, "options":{"wait_for_model":True}})
    vectors = numpy.array(response.json())
    dimension = vectors.shape[1]  
    vectorstore = faiss.IndexFlatL2(dimension) 
    vectorstore.add(vectors)

    return vectorstore



def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id = 'google/flan-t5-xxl', model_kwargs={'temparature':0.5, 'max_length':512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )

    return conversation_chain 
