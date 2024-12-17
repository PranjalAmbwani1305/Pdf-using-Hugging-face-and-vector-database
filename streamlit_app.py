import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import pinecone
import os

# Initialize Pinecone
pinecone.init(api_key='YOUR_PINECONE_API_KEY', environment='YOUR_ENVIRONMENT')

# Create a new index if it doesn't exist
index_name = 'pdf-embeddings'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)  # 384 is the dimension for 'all-MiniLM-L6-v2'

# Connect to the index
index = pinecone.Index(index_name)

# PDF Loader Class
class PDFLoader:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file

    def extract_text(self):
        doc = fitz.open(stream=self.pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

# Embedding Generator Class
class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, text):
        return self.model.encode(text)

# Function to store embeddings in Pinecone
def store_embeddings(embeddings, metadata):
    for i, embedding in enumerate(embeddings):
        index.upsert([(f'doc-{i}', embedding, metadata[i])])

# Streamlit Application
st.title("PDF Embedding Generator")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_loader = PDFLoader(uploaded_file)
    extracted_text = pdf_loader.extract_text()
    
    # Split text into chunks for embedding
    text_chunks = extracted_text.split('\n\n')  # Split by paragraphs or use another method
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(text_chunks)
    
    # Prepare metadata
    metadata = [{'pdf_name': uploaded_file.name, 'chunk_number': i} for i in range(len(embeddings))]
    
    # Store embeddings in Pinecone
    store_embeddings(embeddings, metadata)
    
    st.write("Embeddings generated and stored successfully!")
    st.write(f"Total chunks processed: {len(embeddings)}")
