from langchain_chroma import Chroma

# Load the vector database as retriever from the specified directory
def load_retriever(persist_directory: str, embeddings, collect_name: str, k: int = 20):
    
    # Load the Chroma collection from the specified directory
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collect_name
    )

    # Assign retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    
    # Return the retriever
    return retriever