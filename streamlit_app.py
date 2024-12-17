import streamlit as st
import fitz
import requests
import pinecone
import  sentence_transformers 

# Initialize Pinecone
pinecone.init(api_key="pcsk_6pU2by_7RqfcYiJdc3QoZJVmtqLjBZWZzABszayaXF6fVRJ47pEaKrDu8XZKAsKHZPTrmw", environment="us-east1-gcp")
index_name = "pdf-embeddings"  # Set your Pinecone index name

# Initialize Hugging Face model for embeddings (you can change this to any other suitable model)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create Pinecone index if it doesn't exist
try:
    pinecone.create_index(index_name, dimension=embedding_model.get_sentence_embedding_dimension())
except pinecone.exceptions.ApiException as e:
    if "AlreadyExists" not in str(e):
        raise e

# Connect to the Pinecone index
index = pinecone.Index(index_name)

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
        img_bytes = pix.tobytes()

        st.image(img_bytes, caption=f"Page {page_number}", use_column_width=True)

        # Extract text from the selected page
        page_text = page.get_text()

        # Generate the embedding for the page text
        embedding = embedding_model.encode(page_text)

        # Store the embedding in Pinecone
        # We use the page number as the ID, but you could use any unique identifier
        page_id = str(page_number)
        index.upsert([(page_id, embedding)])

        st.write(f"Embedding for page {page_number} has been stored in Pinecone.")

        # Optionally, display the extracted text for debugging or information
        st.subheader("Extracted Text from Page:")
        st.text(page_text)

        pdf_document.close()

if __name__ == "__main__":
    main()
