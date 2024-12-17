import streamlit as st
import fitz
from pinecone import Client
import PyPDF2
from transformers import pipeline
import numpy as np
from PIL import Image
import io

import os


client = Client(api_key=api_key,"pcsk_6pU2by_7RqfcYiJdc3QoZJVmtqLjBZWZzABszayaXF6fVRJ47pEaKrDu8XZKAsKHZPTrmw")
environment = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")

pinecone.init(api_key=api_key, environment=environment)

index_name = "textembedding"
index = pinecone.Index(index_name)

model_name = "all-MiniLM-L6-v2"
encoder = pipeline("feature-extraction", model=model_name)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def generate_embeddings(text):
    embeddings = encoder(text)[0]
    embeddings = np.mean(embeddings, axis=0)
    return embeddings.tolist()

def store_embeddings_in_pinecone(text, embeddings, doc_id):
    index.upsert([{
        "id": doc_id,
        "vector": embeddings,
        "metadata": {"text": text}
    }])

def search_in_pinecone(query):
    query_embedding = generate_embeddings(query)
    results = index.query(vector=query_embedding, top_k=5)
    return results

def main():
    st.title("PDF Viewer and Embedding App")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()
        pdf_document = fitz.open(stream=pdf_bytes)

        st.write(f"Total Pages: {len(pdf_document)}")

        page_number = st.number_input(
            "Select page number",
            min_value=1,
            max_value=len(pdf_document),
            value=1
        )

        page = pdf_document[page_number - 1]
        pix = page.get_pixmap()

        img = Image.open(io.BytesIO(pix.tobytes("png")))
        st.image(img, caption=f"Page {page_number}", use_column_width=True)

        page_text = page.get_text()

        embedding = generate_embeddings(page_text)

        page_id = str(page_number)
        store_embeddings_in_pinecone(page_text, embedding, page_id)

        st.write(f"Embedding for page {page_number} has been stored in Pinecone.")

        st.subheader("Extracted Text from Page:")
        st.text(page_text)

        pdf_document.close()

if __name__ == "__main__":
    main()
