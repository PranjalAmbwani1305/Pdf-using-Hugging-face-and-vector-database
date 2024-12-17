import pinecone
import streamlit as st

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

index_name = "example-index"

try:
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=128)
        st.info(f"Index '{index_name}' created successfully.")
    else:
        st.info(f"Connected to existing index: '{index_name}'")
    
    index = pinecone.Index(index_name)

except Exception as e:
    st.error(f"Error with Pinecone index: {e}")
    st.stop()

try:
    vector_data = [
        ("vec1", [0.1, 0.2, 0.3, 0.4, 0.5] * 26),
        ("vec2", [0.5, 0.4, 0.3, 0.2, 0.1] * 26)
    ]
    index.upsert(vectors=vector_data)
    st.info(f"Data upserted to index '{index_name}' successfully.")

except Exception as e:
    st.error(f"Error during upsert: {e}")
    st.stop()

try:
    query_vector = [0.1, 0.2, 0.3, 0.4, 0.5] * 26
    results = index.query(
        query_vector=query_vector,
        top_k=2,
        include_metadata=True
    )
    st.write("Query Results:", results)

except Exception as e:
    st.error(f"Error during query: {e}")
    st.stop()
