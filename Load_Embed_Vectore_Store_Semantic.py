from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker

#Load PDF files from directory files are sourced from: https://ncc.abcb.gov.au/editions-national-construction-code
def load_pdf_files(dir):
    loader = PyPDFDirectoryLoader(dir)
    docs = loader.load()
    return docs

#Split the documents into text chunks
def split_documents(docs, embeddings):
    text_splitter = SemanticChunker(embeddings=embeddings)
    texts = text_splitter.split_documents(docs)
    return texts

#Store the text chunks in Chroma
def store_text_chunks(texts, embeddings, persist_directory):    
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name='NCC_codes',
    )
    return vectordb

if __name__ == "__main__":
    dir = "./pdfs"
    embeddings  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    persist_directory = "./chroma"
    docs = load_pdf_files(dir)
    print(f"Loaded {len(docs)} documents")
    texts = split_documents(docs, embeddings=embeddings)
    print(f"Split {len(docs)} documents into {len(texts)} text chunks")
    vectordb = store_text_chunks(texts, embeddings, persist_directory)
    print(f"Chroma directory: {persist_directory}")