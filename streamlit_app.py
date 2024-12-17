import streamlit as st

import PyPDF2
from transformers import pipeline
from tqdm import tqdm
import os

from pinecone import Client
client = Client(api_key= "pcsk_6pU2by_7RqfcYiJdc3QoZJVmtqLjBZWZzABszayaXF6fVRJ47pEaKrDu8XZKAsKHZPTrmw")

index_name = "textembeddings"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)  
index = pinecone.Index(index_name)


embed_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error extracting text: {e}")
    return text

def process_and_store_embeddings(pdf_file, index):
    st.info("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_file)
    if not text:
        st.warning("No text found in the uploaded PDF.")
        return

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    st.success(f"Split text into {len(chunks)} chunks.")

    s
    st.info("Generating embeddings...")
    embeddings = []
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks")):
        embedding = embed_model(chunk, return_tensors="pt")[0].mean(dim=0).tolist()
        embeddings.append((f"chunk-{i}", embedding))

    st.info("Storing embeddings in Pinecone...")
    for chunk_id, embedding in embeddings:
        index.upsert(vectors=[(chunk_id, embedding)])

    st.success("Embeddings stored successfully!")


st.title("PDF Embeddings Generator")
st.write("Upload a PDF file to generate embeddings and store them in Pinecone.")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process PDF and Store Embeddings"):
        with st.spinner("Processing..."):
            process_and_store_embeddings(uploaded_file, index)
