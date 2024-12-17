import streamlit as st
import fitz
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pinecone
import uuid

pinecone.init(api_key='pcsk_6pU2by_7RqfcYiJdc3QoZJVmtqLjBZWZzABszayaXF6fVRJ47pEaKrDu8XZKAsKHZPTrmw', environment='us-east1-gcp')
index_name = 'pdf-embeddings'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)
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

def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()

def store_embeddings_in_pinecone(text, embeddings):
    unique_id = str(uuid.uuid4())
    metadata = {"source": "pdf"}
    index.upsert([(unique_id, embeddings.tolist(), metadata)])

def main():
    st.title("PDF Embeddings with Hugging Face and Pinecone")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.write("Text extracted from PDF:")
        st.text_area("PDF Text", pdf_text, height=200)

        embeddings = generate_embeddings(pdf_text)
        
        store_embeddings_in_pinecone(pdf_text, embeddings)
        st.success("Embeddings successfully stored in Pinecone!")

        st.write("Embedding (first 10 values):", embeddings[:10])

        query_text = st.text_input("Enter query text to search for similar embeddings:")
        
        if query_text:
            query_embedding = generate_embeddings(query_text)
            st.write("Query embedding generated.")
            result = index.query(query_embedding.tolist(), top_k=1)
            st.write(f"Query Result: {result}")

if __name__ == "__main__":
    main()
