import streamlit as st
import pinecone
import PyPDF2
from transformers import pipeline
from tqdm import tqdm
import os

# Step 1: Initialize Pinecone
PINECONE_API_KEY = "your-pinecone-api-key"
pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")  # Update the environment if needed

index_name = "pdf-embeddings"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768)  # dimension matches the Hugging Face model
index = pinecone.Index(index_name)

# Step 2: Hugging Face Embedding Pipeline
embed_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Function to Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error extracting text: {e}")
    return text

# Step 4: Generate Embeddings and Store in Pinecone
def process_and_store_embeddings(pdf_file, index):
    st.info("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_file)
    if not text:
        st.warning("No text found in the uploaded PDF.")
        return

    # Split the text into manageable chunks
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    st.success(f"Split text into {len(chunks)} chunks.")

    # Generate embeddings
    st.info("Generating embeddings...")
    embeddings = []
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks")):
        embedding = embed_model(chunk, return_tensors="pt")[0].mean(dim=0).tolist()
        embeddings.append((f"chunk-{i}", embedding))

    # Store in Pinecone
    st.info("Storing embeddings in Pinecone...")
    for chunk_id, embedding in embeddings:
        index.upsert(vectors=[(chunk_id, embedding)])

    st.success("Embeddings stored successfully!")

# Step 5: Streamlit App Layout
st.title("PDF Embeddings Generator")
st.write("Upload a PDF file to generate embeddings and store them in Pinecone.")

uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process PDF and Store Embeddings"):
        with st.spinner("Processing..."):
            process_and_store_embeddings(uploaded_file, index)
