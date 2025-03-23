from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_community.document_loaders import PyPDFDirectoryLoader

#Load PDF files from directory files are sourced from: https://ncc.abcb.gov.au/editions-national-construction-code
def load_pdf_files(dir):
    loader = PyPDFDirectoryLoader(dir)
    docs = loader.load()
    return docs

#Split the documents into text chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
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
    texts = split_documents(docs)
    print(f"Split {len(texts)} documents into text chunks")
    vectordb = store_text_chunks(texts, embeddings, persist_directory)
    print(f"Chroma directory: {persist_directory}")