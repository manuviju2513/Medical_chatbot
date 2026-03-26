from pinecone import ServerlessSpec,Pinecone
from langchain_pinecone import PineconeVectorStore
from src.helper import *
from dotenv import load_dotenv
import os
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_data = load_pdf_files("data/")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(minimal_docs)

embeddings = download_embedding()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name= index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")

    )
index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name

) 

