import os
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import streamlit as st
from pinecone import ServerlessSpec

# Access API keys securely from Streamlit secrets
HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone client using the correct initialization method
pinecone.init(api_key=PINECONE_API_KEY, environment="us-east-1")  # Updated to us-east-1 region

# Chatbot Class
class GPMCChatbot:
    def __init__(self):
        # Load documents from the PDF file
        pdf_loader = PyMuPDFLoader("gpmc.pdf")  # Ensure the file exists in the root directory
        raw_documents = pdf_loader.load()

        # Split documents into smaller chunks for processing
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        self.document_chunks = splitter.split_documents(raw_documents)

        # Initialize embeddings model
        self.embeddings_model = HuggingFaceEmbeddings()

        # Define the index name and create it if necessary
        index_name = "chatbot"  # Updated index name

        # Create Pinecone index if it does not exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=768,  # Adjust based on your embedding model
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Updated to us-east-1 region
            )

        # Initialize the Pinecone vector store with the index
        self.vector_store = Pinecone.from_documents(
            documents=self.document_chunks,
            embeddings=self.embeddings_model,
            index_name=index_name
        )

        # Define Hugging Face model for generating responses
        self.llm = HuggingFaceHub(
            repo_id="tiiuae/falcon-7b-instruct",  # Updated Hugging Face model
            api_token=HUGGINGFACE_API_KEY,
            model_kwargs={"temperature": 0.6, "top_k": 40}
        )

        # Define the prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template=(
                "You are a helpful assistant specializing in GPMC."
                "\nContext: {context}\n"
                "User Query: {query}\n\n"
                "Response:"
            )
        )

    def get_response(self, query):
        # Retrieve relevant documents
        retriever = self.vector_store.as_retriever()
        context = retriever.retrieve(query)

        # Generate the response using the LLM
        formatted_prompt = self.prompt_template.format(context=context, query=query)
        response = self.llm.generate([formatted_prompt])[0]

        return response

# Streamlit app setup
st.set_page_config(page_title="Chatbot for GPMC Assistance")  # Updated title

# Sidebar configuration
st.sidebar.title("Chatbot")
st.sidebar.info("Ask questions about GPMC.")

# Chatbot instance initialization
@st.cache_resource
def initialize_chatbot():
    return GPMCChatbot()

# Main application logic
chatbot = initialize_chatbot()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "Hello! How can I assist you with the GPMC today?"}]

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Process user input
user_input = st.chat_input("Type your query here...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Fetching response..."):
            try:
                bot_response = chatbot.get_response(user_input)
            except Exception as e:
                bot_response = f"An error occurred: {str(e)}"
            st.write(bot_response)
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
