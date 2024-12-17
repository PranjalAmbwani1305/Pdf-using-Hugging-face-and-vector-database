import PyPDF2
from transformers import pipeline
import pinecone
import streamlit as st

api_key = "pcsk_6pU2by_7RqfcYiJdc3QoZJVmtqLjBZWZzABszayaXF6fVRJ47pEaKrDu8XZKAsKHZPTrmw"
environment = "us-east1-gcp"

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
    return embeddings

def store_embeddings_in_pinecone(text, embeddings):
    index.upsert([
        {
            "id": "document_id", 
            "vector": embeddings,
            "metadata": {
                "text": text
            }
        }
    ])

def search_in_pinecone(query):
    query_embedding = generate_embeddings(query)
    results = index.query(vector=query_embedding, top_k=5)
    return results

def main():
    st.title("PDF Search")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        pdf_embeddings = generate_embeddings(pdf_text)
        store_embeddings_in_pinecone(pdf_text, pdf_embeddings)
        st.success("PDF uploaded and indexed successfully!")

    query = st.text_input("Enter your query")
    if query:
        results = search_in_pinecone(query)
        for result in results['matches']:
            st.write(result['metadata']['text'])

if __name__ == "__main__":
    main()
