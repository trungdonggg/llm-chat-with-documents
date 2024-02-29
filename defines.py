import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma




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
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunked_text = text_splitter.split_text(raw_text) 

    return chunked_text



def get_vectorstore(chunked_text):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(chunked_text, embedding_function)

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




def handle_user_input(conv, promt):
    response = conv({'question': promt})
    st.write(response)

