import streamlit as st
import fitz 
import pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import uuid

pinecone_api_key = st.secrets["pinecone"]["api_key"]
pinecone.init(api_key=pinecone_api_key, environment="us-east1-gcp")

index_name = "pdf-index"
dimension = 384
metric = "cosine"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=dimension, metric=metric)

index = pinecone.Index(index_name)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def upsert_to_pinecone(text, embeddings, metadata=None):
    id = str(uuid.uuid4())
    metadata = metadata or {"source": "uploaded_pdf"}
    vector = {
        "id": id,
        "values": embeddings.tolist(),
        "metadata": metadata
    }
    index.upsert(vectors=[vector])

def process_pdf(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    embeddings = generate_embedding(text)
    metadata = {"source": "uploaded_pdf"}
    upsert_to_pinecone(text, embeddings, metadata)
    st.success(f"Successfully upserted the PDF to Pinecone.")

def query_pinecone(query_text, top_k=5):
    query_embedding = generate_embedding(query_text)
    query_result = index.query(
        queries=[query_embedding.tolist()],
        top_k=top_k,
        include_metadata=True
    )
    return query_result

st.title("PDF Embedding")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    st.write(f"File uploaded: {uploaded_file.name}")
    if st.button("Process PDF and Store in Pinecone"):
        process_pdf(uploaded_file)

query_text = st.text_input("Enter a query text to search for similar PDFs in Pinecone:")
if query_text:
    results = query_pinecone(query_text)
    if results['matches']:
        st.write("### Search Results:")
        for match in results['matches']:
            st.write(f"**ID:** {match['id']}, **Score:** {match['score']}")
            st.write(f"**Metadata:** {match['metadata']}")
    else:
        st.write("No matches found.")
