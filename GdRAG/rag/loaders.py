from typing import List
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
import os

def load_text_files(dir_path: str) -> List[Document]:
    docs = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(".txt"):
                path = os.path.join(root, f)
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
            elif f.lower().endswith(".pdf"):
                path = os.path.join(root, f)
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
    return docs

# small helper to chunk (LangChain has RecursiveCharacterTextSplitter)
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents: List[Document], chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    new_docs = []
    for d in documents:
        texts = splitter.split_text(d.page_content)
        for i, t in enumerate(texts):
            md = d.metadata.copy() if d.metadata else {}
            md.update({"source": d.metadata.get("source", "") if d.metadata else ""})
            new_docs.append(Document(page_content=t, metadata=md))
    return new_docs
