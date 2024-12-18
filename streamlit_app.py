import streamlit as st
import fitz  
from sentence_transformers import SentenceTransformer
import pinecone
import os

os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="us-west1-gcp")

index_name = "pdf-embeddings"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768)

index = pinecone.Index(index_name)

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
    for i, embedding in enumerate(embeddings):
        id = f'doc-{i}'
        index.upsert([(id, embedding, metadata[i])])

st.title("PDF Embedding Generator")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_loader = PDFLoader(uploaded_file)
    extracted_text = pdf_loader.extract_text()
    
    text_chunks = extracted_text.split('\n\n')
    
    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(text_chunks)
    
    metadata = [{'pdf_name': uploaded_file.name, 'chunk_number': i} for i in range(len(embeddings))]
    
    store_embeddings(embeddings, metadata)
    
    st.write("Embeddings generated and stored successfully!")
    st.write(f"Total chunks processed: {len(embeddings)}")
