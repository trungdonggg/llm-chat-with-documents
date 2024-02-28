import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv
load_dotenv()
import requests
import faiss
import numpy

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
    llm = None
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )

    return conversation_chain 


# text = '''Datasets is a library for quickly accessing and sharing datasets. Let's host the embeddings dataset in the Hub using the user interface (UI). Then, anyone can load it with a single line of code. You can also use the terminal to share datasets; see the documentation for the steps. In the notebook companion of this entry, you will be able to use the terminal to share the dataset. If you want to skip this section, check out the ITESM/embedded_faqs_medicare repo with the embedded FAQs.

# First, we export our embeddings from a Pandas DataFrame to a CSV. You can save your dataset in any way you prefer, e.g., zip or pickle; you don't need to use Pandas or CSV. Since our embeddings file is not large, we can store it in a CSV, which is easily inferred by the datasets.load_dataset() function we will employ in the next section (see the Datasets documentation), i.e., we don't need to create a loading script. We will save the embeddings with the name embeddings.csv. Datasets is a library for quickly accessing and sharing datasets. Let's host the embeddings dataset in the Hub using the user interface (UI). Then, anyone can load it with a single line of code. You can also use the terminal to share datasets; see the documentation for the steps. In the notebook companion of this entry, you will be able to use the terminal to share the dataset. If you want to skip this section, check out the ITESM/embedded_faqs_medicare repo with the embedded FAQs.

# First, we export our embeddings from a Pandas DataFrame to a CSV. You can save your dataset in any way you prefer, e.g., zip or pickle; you don't need to use Pandas or CSV. Since our embeddings file is not large, we can store it in a CSV, which is easily inferred by the datasets.load_dataset() function we will employ in the next section (see the Datasets documentation), i.e., we don't need to create a loading script. We will save the embeddings with the name embeddings.csv.'''

# ct = get_text_chunked(text)
# v = get_vectorstore(ct)
# print(v)