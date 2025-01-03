import streamlit as st
import fitz  
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone as PineconeClient
import os
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import time
import pytesseract
from PIL import Image
import io

os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

class PDFLoader:
    def __init__(self, pdf_file):
        if pdf_file is None:
            raise ValueError("PDF file is not provided.")
        self.pdf_file = pdf_file
        self.extracted_text = self.extract_text()

        text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=4)
        self.docs = text_splitter.split_documents([Document(page_content=self.extracted_text)])

        self.index_name = "textembedding"
        self.pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY'))

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=384,
                metric='cosine',
            )
            st.write(f"Index '{self.index_name}' created.")
        else:
            st.write(f"Index '{self.index_name}' already exists.")

        self.index = self.pc.Index(self.index_name)

    def extract_text(self):
        try:
            if not self.pdf_file:
                raise ValueError("The PDF file is empty.")

            doc = fitz.open(stream=self.pdf_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text("text")

            if text.strip():
                return text
            else:
                st.warning("No selectable text found. Using OCR to extract text from images.")
                return self.extract_text_with_ocr(doc)
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {str(e)}")

    
    def extract_text_from_image(self, img):
        try:
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            st.error(f"Error during OCR: {str(e)}")
            return ""


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
            st.write(f"Upsert response: {response}")

            if response.get("upserted", 0) > 0:
                st.write(f"Successfully upserted {response['upserted']} vectors.")
            else:
                st.error("No vectors were upserted.")
            break
        except Exception as e:
            st.error(f"Error during Pinecone upsert: {str(e)}")
            if attempt < retries - 1:
                st.write(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.error("Max retries reached. Please check the service status.")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    loader = PDFLoader(uploaded_file)
    text_chunks = [doc.page_content for doc in loader.docs]

    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(text_chunks)

    st.write(f"Generated embeddings: {embeddings[:2]}")
    st.write(f"Shape of embeddings: {np.array(embeddings).shape}")

    metadata = [{"chunk_index": i, "source": "uploaded_pdf"} for i in range(len(embeddings))]

    store_embeddings(loader.index, embeddings, metadata)              
