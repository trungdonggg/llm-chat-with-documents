import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.vectorstores import FAISS  
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# (langchain 0.1.0)




def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()

    return text



def get_text_chunked(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunked_text = text_splitter.split_text(raw_text) 

    return chunked_text



def get_vectorstore(chunked_text):
    embeddings = HuggingFaceHubEmbeddings(repo_id='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(texts=chunked_text, embedding=embeddings)

    return vectorstore



def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain




def handle_user_input(conv, promt):
    response = st.session_state.conversation({'question': promt})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)
            



