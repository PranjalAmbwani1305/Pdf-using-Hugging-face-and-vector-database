import streamlit as st
import fitz  
from sentence_transformers import SentenceTransformer
import pinecone
import os
import numpy as np
import easyocr


os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

index_name = "textembedding" 
if index_name not in pinecone.list():
    pinecone.create_index(index_name, dimension=384)

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
    stored_data = []  
    upsert_data = [] 
    for i, embedding in enumerate(embeddings):
        if isinstance(embedding, np.ndarray):  
            embedding = embedding.tolist()

        id = f'doc-{i}'
        metadata_dict = metadata[i] if isinstance(metadata[i], dict) else {}

        if index is not None:
            upsert_data.append((id, embedding, metadata_dict))
        else:
            stored_data.append((id, embedding, metadata_dict)) 

    if index is not None:
        try:
            response = index.upsert(vectors=upsert_data)
            if response.get("upserted", 0) > 0:
                print(f"Successfully upserted {response['upserted']} vectors.")
                st.write(f"Successfully upserted {response['upserted']} vectors.")
        except Exception as e:
            print(f"Error during Pinecone upsert: {str(e)}")
            st.error(f"Error during Pinecone upsert: {str(e)}")
    else:
        # Store locally if Pinecone isn't available
        print(f"Stored {len(stored_data)} embeddings locally.")
        st.write(f"Stored {len(stored_data)} embeddings locally.")
        return stored_data

st.title("PDF Generator")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_loader = PDFLoader(uploaded_file)
    extracted_text = pdf_loader.extract_text()

    text_chunks = extracted_text.split('\n\n')

    if not text_chunks:
        st.error("No text chunks to process.")
    else:
        embedding_generator = EmbeddingGenerator()
        embeddings = embedding_generator.generate_embeddings(text_chunks)

        metadata = [{'pdf_name': uploaded_file.name, 'chunk_number': i} for i in range(len(embeddings))]

        store_embeddings(embeddings, metadata)

        st.write("Embeddings generated and stored successfully!")
        st.write(f"Total chunks processed: {len(embeddings)}")
