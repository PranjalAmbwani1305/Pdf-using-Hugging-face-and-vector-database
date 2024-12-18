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

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  
                metric='cosine',
            )
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

def store_embeddings(index, embeddings, metadata):
    upsert_data = [] 
    for i, embedding in enumerate(embeddings):
        if isinstance(embedding, np.ndarray):  
            embedding = embedding.tolist()

        id = f'doc-{i}'
        metadata_dict = metadata[i] if isinstance(metadata[i], dict) else {}

        upsert_data.append((id, embedding, metadata_dict))

    try:
        response = index.upsert(vectors=upsert_data)
        if response.get("upserted", 0) > 0:
            st.write(f"Successfully upserted {response['upserted']} vectors.")
    except Exception as e:
        st.error(f"Error during Pinecone upsert: {str(e)}")

st.title("PDF embedding using Hugging Face and store in Pinecone")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    try:
        file_size = uploaded_file.size
        if file_size == 0:
            st.error("The uploaded PDF file is empty.")
        else:
            st.write(f"File size: {file_size / 1024:.2f} KB")

            pdf_loader = PDFLoader(uploaded_file)
            extracted_text = pdf_loader.extracted_text

            # Use the already split documents
            text_chunks = [doc.page_content for doc in pdf_loader.docs]

            if not text_chunks:
                st.error("No text chunks to process.")
            else:
                embedding_generator = EmbeddingGenerator()
                embeddings = embedding_generator.generate_embeddings(text_chunks)

                metadata = [{'pdf_name': uploaded_file.name, 'chunk_number': i} for i in range(len(embeddings))]

                store_embeddings(pdf_loader.index, embeddings, metadata)

                st.write("Embeddings generated and stored successfully!")
                st.write(f"Total chunks processed: {len(embeddings)}")

    except Exception as e:
        st.error(f"
