from langchain.vectorstores import Vectorizer
import numpy 
import requests
import os
from dotenv import load_dotenv
load_dotenv()

class CustomEmbeddings(Vectorizer):
    def __init__(self):
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {self.hf_token}"}

    def embed(self, chunked_text):
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": chunked_text, "options":{"wait_for_model":True}})
        embeddings = numpy.array(response.json())
        return embeddings

  
