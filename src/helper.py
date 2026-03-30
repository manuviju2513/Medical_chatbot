from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    document = loader.load()
    return document

def filter_to_minimal_docs(docs:List[Document])-> List[Document]:
    """ List of document containing Source and content only"""
    minimal_docs: List[Document] =[]
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content= doc.page_content,
                metadata= {"source": src}
            )
        )

    return minimal_docs


def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap= 20
    )
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks




def download_embedding():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name = model_name
    )
    return embeddings

