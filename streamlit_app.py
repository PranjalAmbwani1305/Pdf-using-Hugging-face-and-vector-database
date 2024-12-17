import pinecone
import streamlit as st
import PyPDF2
import sentence_transformers
import huggingface_hub
from sentence_transformers import SentenceTransformer

pinecone_api_key = st.secrets["PINECONE_API_KEY"]

try:
    pinecone.init(
        api_key=pinecone_api_key,
        environment="us-east1-gcp"
    )
    st.success("Pinecone initialized successfully!")

except Exception as e:
    st.error(f"Failed to initialize Pinecone: {e}")
    st.stop()

index_name = "textembedding"

try:
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=768)
        st.info(f"Index '{index_name}' created successfully.")
    else:
        st.info(f"Connected to existing index: '{index_name}'")
    
    index = pinecone.Index(index_name)

except Exception as e:
    st.error(f"Error with Pinecone index: {e}")
    st.stop()

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_pdf_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    pdf_text = extract_pdf_text(uploaded_file)
    if pdf_text:
        st.write("Extracted Text from PDF:")
        st.write(pdf_text[:1000])

        embeddings = model.encode([pdf_text])

        try:
            vector_data = [
                ("pdf_vector_1", embeddings[0])
            ]
            index.upsert(vectors=vector_data)
            st.info("PDF embeddings upserted to Pinecone successfully.")

        except Exception as e:
            st.error(f"Error during upsert: {e}")
            st.stop()

        try:
            query_vector = model.encode(["sample query text to search"])
            results = index.query(
                query_vector=query_vector[0],
                top_k=1,
                include_metadata=True
            )
            st.write("Query Results:", results)

        except Exception as e:
            st.error(f"Error during query: {e}")
            st.stop()
    else:
        st.error("No text could be extracted from the PDF.")
