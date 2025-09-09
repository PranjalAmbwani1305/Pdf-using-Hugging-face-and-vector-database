import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone as PineconeClient
import os
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import time
import pytesseract
from PIL import Image
import io

# Load API keys from Streamlit secrets
os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

class PDFLoader:
    def __init__(self, pdf_file):
        if pdf_file is None:
            raise ValueError("PDF file is not provided.")
        self.pdf_file = pdf_file
        self.extracted_text = self.extract_text()

        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        self.docs = text_splitter.split_documents([Document(page_content=self.extracted_text)])

        self.index_name = "textembedding"
        self.pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY'))

        indexes = [idx['name'] for idx in self.pc.list_indexes()]
        if self.index_name not in indexes:
            self.pc.create_index(name=self.index_name, dimension=384, metric='cosine')
            st.write(f"✅ Created Pinecone index: `{self.index_name}`")
        else:
            st.write(f"ℹ️ Pinecone index `{self.index_name}` already exists.")

        self.index = self.pc.Index(self.index_name)

    def extract_text(self):
        try:
            doc = fitz.open(stream=self.pdf_file.read(), filetype="pdf")
            text = "".join(page.get_text("text") for page in doc)

            if text.strip():
                return text
            else:
                st.warning("No selectable text found. Using OCR...")
                return self.extract_text_with_ocr(doc)
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {str(e)}")

    def extract_text_with_ocr(self, doc):
        text = ""
        for page_index in range(len(doc)):
            pix = doc[page_index].get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text += self.extract_text_from_image(img)
        return text

    def extract_text_from_image(self, img):
        try:
            return pytesseract.image_to_string(img)
        except Exception as e:
            st.error(f"OCR error: {str(e)}")
            return ""


class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, text_chunks):
        return np.array(self.model.encode(text_chunks, show_progress_bar=True))


def store_embeddings(index, embeddings, metadata, batch_size=100, retries=3, delay=2):
    upsert_data = []
    for i, embedding in enumerate(embeddings):
        upsert_data.append((f"doc-{i}", embedding.tolist(), metadata[i]))

    st.write(f"📦 Preparing to upsert {len(upsert_data)} vectors in batches of {batch_size}.")

    for batch_start in range(0, len(upsert_data), batch_size):
        batch = upsert_data[batch_start: batch_start + batch_size]
        for attempt in range(retries):
            try:
                response = index.upsert(vectors=batch)
                st.write(f"✅ Upserted batch {batch_start // batch_size + 1}, response: {response}")
                break
            except Exception as e:
                wait_time = delay * (2 ** attempt)
                st.error(f"⚠️ Error during Pinecone upsert: {str(e)}")
                if attempt < retries - 1:
                    st.write(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    st.error("🚨 Max retries reached. Skipping this batch.")


# Streamlit UI
st.title("📄 PDF to Pinecone Embedding Uploader")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    loader = PDFLoader(uploaded_file)
    text_chunks = [doc.page_content for doc in loader.docs]

    embedding_generator = EmbeddingGenerator()
    embeddings = embedding_generator.generate_embeddings(text_chunks)

    st.write(f"🔢 Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    st.write("🧾 Example embedding vector:", embeddings[0][:10])

    metadata = [{"chunk_index": i, "source": "uploaded_pdf"} for i in range(len(text_chunks))]

    store_embeddings(loader.index, embeddings, metadata)
    st.success("🎉 All embeddings uploaded successfully!")
