import streamlit as st
import fitz  
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone as PineconeClient, ServerlessSpec 
import os
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document


os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

class PDFLoader:
    def __init__(self, pdf_file):
        if pdf_file is None:
            raise ValueError("PDF file is not provided.")
        self.pdf_file = pdf_file
        self.extracted_text = self.extract_text()
        
        # Create Document objects for each chunk of text
        text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=4)
        self.docs = text_splitter.split_documents([Document(page_content=self.extracted_text)])

        self.index_name = "textembedding"
        self.pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY')) 

        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  
                metric='cosine',
            )
            st.write(f"Index '{self.index_name}' created.")
        else:
            st.write(f"Index '{self.index_name}' already exists.")
        
        self.index = self.pc.Index(self.index_name)  # Initialize the index

    def extract_text(self):
        try:
            if not self.pdf_file:
                raise ValueError("The PDF file is empty.")
            
            doc = fitz.open(stream=self.pdf_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {str(e)}")

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, text_chunks):
        return self.model.encode(text_chunks)

def store_embeddings(index, embeddings, metadata, retries=3, delay=2):
    upsert_data = [] 
    for i, embedding in enumerate(embeddings):
        if isinstance(embedding, np.ndarray):  
            embedding = embedding.tolist()

        id = f'doc-{i}'
        metadata_dict = metadata[i] if isinstance(metadata[i], dict) else {}

        upsert_data.append((id, embedding, metadata_dict))

    st.write(f"Preparing to upsert {len(upsert_data)} vectors.")
    
    for attempt in range(retries):
        try:
            response = index.upsert(vectors=upsert_data)
            st.write(f"Upsert response: {response}")  # Log the response
            if response.get("upserted", 0) > 0:
                st.write(f"Successfully upserted {response['upserted']} vectors.")
            else:
                st.error("No vectors were upserted.")
            break  # Exit the loop if successful
        except Exception as e:
            st.error(f"Error during Pinecone upsert: {str(e)}")
            if attempt < retries - 1:  # If not the last attempt
                st.write(f"Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
            else:
                st.error("Max retries reached. Please check the service status.")
