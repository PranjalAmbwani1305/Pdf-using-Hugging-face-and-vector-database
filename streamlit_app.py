import streamlit as st
import fitz
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pinecone
import uuid
import os

os.environ["PINECONE_API_KEY"] = "your-pinecone-api-key-here"

try:
    pinecone.init()
    print("Pinecone initialized successfully.")
except Exception as e:
    st.error(f"Error initializing Pinecone: {e}")
    exit()

index_name = 'textembedding'
dimension = 1536

try:
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension)
        print(f"Index '{index_name}' created successfully.")
except Exception as e:
    st.error(f"Error creating or accessing index: {e}")
    exit()

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

def store_embeddings_in_pinecone(embeddings):
    unique_id = str(uuid.uuid4())
    metadata = {"source": "pdf"}
    try:
        index.upsert([(unique_id, embeddings.tolist(), metadata)])
        return unique_id
    except Exception as e:
        st.error(f"Error storing embeddings in Pinecone: {e}")
        return None

def main():
    st.title("PDF Embeddings")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        try:
            pdf_text = extract_text_from_pdf(uploaded_file)
            st.write("Text extracted from PDF:")
            st.text_area("PDF Text", pdf_text, height=200)

            embeddings = generate_embeddings(pdf_text)
            st.write("Embedding generated. (First 10 values):")
            st.write(embeddings[:10])

            unique_id = store_embeddings_in_pinecone(embeddings)
            if unique_id:
                st.success("Embeddings successfully stored in Pinecone!")

            query_text = st.text_input("Enter query text to search for similar embeddings:")
            if query_text:
                query_embedding = generate_embeddings(query_text)
                st.write("Query embedding generated.")
                
                try:
                    result = index.query(query_embedding.tolist(), top_k=1)
                    st.write(f"Query result: {result}")
                except Exception as e:
                    st.error(f"Error querying Pinecone: {e}")
if __name__ == "__main__":
    main()
