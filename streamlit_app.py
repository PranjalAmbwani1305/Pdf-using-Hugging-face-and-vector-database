import streamlit as st
import fitz  
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone as PineconeClient, ServerlessSpec 
import os
import numpy as np
import easyocr


os.environ['HUGGINGFACE_API_KEY'] = st.secrets["HUGGINGFACE_API_KEY"]
os.environ['PINECONE_API_KEY'] = st.secrets["PINECONE_API_KEY"]

index = None

class PDFLoader:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file
        
        # Split documents into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=4)
        self.docs = text_splitter.split_documents(documents)

        
        self.embeddings = HuggingFaceEmbeddings()

        # Define the index name
        self.index_name = "textembedding"

        # Initialize Pinecone client
        self.pc = PineconeClient(api_key=os.getenv('PINECONE_API_KEY')) 
     
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  
                metric='cosine',
            )
            spec=ServerlessSpec(
                    cloud='aws', 
                    region='us-east-1'  
                )
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id, 
            temperature=0.8, 
            top_k=50, 
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )

class PDFLoader:
    def __init__(self, pdf_file):
        self.pdf_file = pdf_file

    def extract_text(self):
        doc = fitz.open(stream=self.pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, text_chunks):
        return self.model.encode(text_chunks)

def store_embeddings(embeddings, metadata):
    stored_data = []  
    upsert_data = [] 
    for i, embedding in enumerate(embeddings):
        if isinstance(embedding, np.ndarray):  
            embedding = embedding.tolist()

        id = f'doc-{i}'
        metadata_dict = metadata[i] if isinstance(metadata[i], dict) else {}

        if index is not None:
            upsert_data.append((id, embedding, metadata_dict))
        else:
            stored_data.append((id, embedding, metadata_dict)) 

    if index is not None:
        try:
            response = index.upsert(vectors=upsert_data)
            if response.get("upserted", 0) > 0:
                print(f"Successfully upserted {response['upserted']} vectors.")
                st.write(f"Successfully upserted {response['upserted']} vectors.")
        except Exception as e:
            print(f"Error during Pinecone upsert: {str(e)}")
            st.error(f"Error during Pinecone upsert: {str(e)}")
    else:
        # Store locally if Pinecone isn't available
        print(f"Stored {len(stored_data)} embeddings locally.")
        st.write(f"Stored {len(stored_data)} embeddings locally.")
        return stored_data

st.title("PDF embedding using hugging face and store in pinecone")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_loader = PDFLoader(uploaded_file)
    extracted_text = pdf_loader.extract_text()

    text_chunks = extracted_text.split('\n\n')

    if not text_chunks:
        st.error("No text chunks to process.")
    else:
        embedding_generator = EmbeddingGenerator()
        embeddings = embedding_generator.generate_embeddings(text_chunks)

        metadata = [{'pdf_name': uploaded_file.name, 'chunk_number': i} for i in range(len(embeddings))]

        store_embeddings(embeddings, metadata)

        st.write("Embeddings generated and stored successfully!")
        st.write(f"Total chunks processed: {len(embeddings)}")
