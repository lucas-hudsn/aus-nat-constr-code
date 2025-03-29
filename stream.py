import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import getpass
import os   
from Vectore_Fns import load_retriever, generate_answer_from_context

embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
persist_directory = "./chroma"
collect_name='NCC_codes'
load_dotenv()

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
    assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key

llm = init_chat_model("deepseek-ai/deepseek-r1", model_provider="nvidia")

retriever = load_retriever(persist_directory, embeddings, collect_name, 3)

st.title("Australian NCC Codes Chatbot with R1")
st.write("Ask any question, and the chatbot will respond using context from the vector database!")

user_query = st.text_input("Enter your question here:")

if st.button("Get Response"):
    with st.spinner("Generating response..."):
        try:
            st.write(doc.page_content for doc in retriever.invoke(user_query))
            st.success("Response:")
            st.write_stream(generate_answer_from_context(retriever, llm, user_query))
        except Exception as e:
            st.error(f"An error occurred: {e}")