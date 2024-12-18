import time

def store_embeddings(index, embeddings, metadata, retries=3, delay=2):
    upsert_data = [] 
    for i, embedding in enumerate(embeddings):
        if isinstance(embedding, np.ndarray):  
            embedding = embedding.tolist()

        id = f'doc-{i}'
        metadata_dict = metadata[i] if isinstance(metadata[i], dict) else {}

        upsert_data.append((id, embedding, metadata_dict))

    st.write(f"Preparing to upsert {len(upsert_data)} vectors.")
    
    for attempt in range(retries):
        try:
            response = index.upsert(vectors=upsert_data)
            st.write(f"Upsert response: {response}")  # Log the response
            if response.get("upserted", 0) > 0:
                st.write(f"Successfully upserted {response['upserted']} vectors.")
            else:
                st.error("No vectors were upserted.")
            break  # Exit the loop if successful
        except Exception as e:
            st.error(f"Error during Pinecone upsert: {str(e)}")
            if attempt < retries - 1:  # If not the last attempt
                st.write(f"Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
            else:
                st.error("Max retries reached. Please check the service status.")
