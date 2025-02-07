from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Calling the URL and make it in multiple chunk
def url_call(url):
    loader = UnstructuredURLLoader(urls=[url])
    document = loader.load()
    docs = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
    ).split_documents(document)

    return docs

# Databse creation and save the db file

def db_store(docs):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_documents(docs, embedding)
    db.save_local("faiss_db")
    return db