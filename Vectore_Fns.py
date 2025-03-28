from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load the vector database as retriever from the specified directory
def load_retriever(persist_directory: str, embeddings, collect_name: str, k: int = 3):
    
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


def generate_answer_from_context(retriever, llm, question: str):

    # Define the message template for the prompt
    message = """
    Answer this question using the provided context only.
    {question}

    Context:
    {context}
    """

    # Create a chat prompt template from the message
    prompt = ChatPromptTemplate.from_messages([("human", message)])

    # Create a RAG (Retrieval-Augmented Generation) chain
    # This chain retrieves context, passes through the question,
    # formats the prompt, and generates an answer using the language model
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

    # Invoke the RAG chain with the question and return the generated content
    return rag_chain.invoke(question)#.content