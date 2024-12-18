import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import pinecone
import os
import numpy as np

# Set environment variables
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="us-west1-gcp")

# Define index name
index_name = "textembedding"

# Check if index exists, create it if not
if index_name not in pinecone.list_indexes():
    print("Index does not exist. Creating index...")
    pinecone.create_index(index_name, dimension=768)  # 768 is the dimension for 'all-MiniLM-L6-v2'

# Connect to Pinecone index (Create index object here)
index = pinecone.Index(index_name)

# Debugging: Check if the index object is properly initialized
print(f"Index object: {index}")
print(f"Type of index: {type(index)}")

class PDFLoader:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file

    def extract_text(self):
        doc = fitz.open(stream=self.pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)  

    def generate_embeddings(self, text_chunks):
        return self.model.encode(text_chunks)  

def store_embeddings(embeddings, metadata):
    upsert_data = []
    for i, embedding in enumerate(embeddings):
        if isinstance(embedding, np.ndarray):  # Convert numpy array to list if needed
            embedding = embedding.tolist()

        # Create a unique ID for each embedding
        id = f'doc-{i}'

        # Ensure metadata is a dictionary
        metadata_dict = metadata[i] if isinstance(metadata[i], dict) else {}

        # Prepare upsert data
        upsert_data.append((id, embedding, metadata_dict))
    
    # Debugging: Check the data before upserting
    print(f"Upsert data: {upsert_data}")
    print(f"Type of upsert data: {type(upsert_data)}")

    try:
        # Check the index is still accessible
        print("Attempting to upsert data...")
        response = index.upsert(upsert_data)  # This is where the actual upsert happens
        print(f"Successfully upserted {len(upsert_data)} items. Response: {response}")
    except Exception as e:
        print(f"Error during upsert: {str(e)}")
        raise

st.title("PDF Embedding Generator")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    pdf_loader = PDFLoader(uploaded_file)
    extracted_text = pdf_loader.extract_text()

    # Split text into chunks (for large texts)
    text_chunks = extracted_text.split('\n\n')  # Split by double newline as an example

    # Generate embeddings for the chunks of text
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(text_chunks)

    # Create metadata for each chunk
    metadata = [{'pdf_name': uploaded_file.name, 'chunk_number': i} for i in range(len(embeddings))]

    # Store embeddings in Pinecone
    store_embeddings(embeddings, metadata)

    # Feedback to the user
    st.write("Embeddings generated and stored successfully!")
    st.write(f"Total chunks processed: {len(embeddings)}")
