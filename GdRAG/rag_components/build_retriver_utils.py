from typing import List, Dict
from chardet.universaldetector import UniversalDetector
import os
import torch
import shutil
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from configs import BaseConfig
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter


def get_retrieval_info(cfg: BaseConfig):
    """
    Get retrieval information from the configuration.
    """
    retrieval_name = '_'.join(cfg.datastorage.raw_data_dir)
    if len(cfg.datastorage.raw_data_dir) != 1:
        retrieval_name = 'mix_' + retrieval_name

    retrieval_store_path = f"./retrieval_stores/{retrieval_name}/{cfg.embedding.model_name}/{cfg.datastorage.tool}"
    return retrieval_name, retrieval_store_path


def get_encoding_of_file(path: str) -> str:
    """
    return the encoding of a file
    """
    detector = UniversalDetector()
    with open(path, 'rb') as file:
        data = file.readlines()
        for line in data:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    return detector.result['encoding']

def load_files2docs(dir_path: str) -> List[Document]:
    '''
    Load text and PDF files from a directory into a list of Document objects.
    '''
    docs = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(".txt"):
                path = os.path.join(root, f)
                encoding = get_encoding_of_file(path)
                loader = TextLoader(path, encoding=encoding)
                docs.extend(loader.load())
            elif f.lower().endswith(".pdf"):
                path = os.path.join(root, f)
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
    print(f'File number of {dir_path}: {len(docs)}')
    return docs


class SingleFileSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        return [text]

class LineBreakTextSplitter(TextSplitter):
    def split_text(self, text: str) -> List[str]:
        return text.split("\n\n")


def chunk_documents(documents: List[Document], cfg: BaseConfig):
    '''
    Chunk documents into smaller pieces based on the configuration.
    '''
    def get_splitter(cfg: BaseConfig):
        if cfg.chunk.method == "by_single_file":
            return SingleFileSplitter()
        elif cfg.chunk.method == "by_two_line_breaks":
            return LineBreakTextSplitter()
        elif cfg.chunk.method == 'recursive':
            return RecursiveCharacterTextSplitter(
                        chunk_size=cfg.chunk.params.get("chunk_size", 1000), # get chunk_size, default 1000
                        chunk_overlap=cfg.chunk.params.get("chunk_overlap", 200), # get chunk_overlap, default 200
                    )
        
    splitter = get_splitter(cfg)

    split_docs = splitter.split_documents(documents)

    return split_docs


def get_data_chunks(cfg: BaseConfig):
    # load raw data
    docs = []
    for data_path in cfg.datastorage.raw_data_dir:
        docs.extend(load_files2docs(data_path))

    # get data chunk
    chunk_docs = chunk_documents(docs, cfg)

    return chunk_docs


NER_MODELS = {
    "medical": {
        "model": "d4data/biomedical-ner-all",
        "entities": ["DISEASE", "CHEMICAL", "GENE", "PROTEIN", "CELL_TYPE", "CELL_LINE"]
    },
    "finance": {
        "model": "jupup/finbert-ner",
        "entities": ["COMPANY", "ORG", "MONEY", "STOCK", "CURRENCY", "PERCENT"]
    },
    "science": {
        "model": "allenai/scibert_scivocab_cased",
        "entities": ["METHOD", "MATERIAL", "TASK", "METRIC", "GENERIC"]  # 依赖具体finetune
    },
    "legal": {
        "model": "nlpaueb/legal-bert-base-uncased",
        "entities": ["LAW", "COURT", "CASE", "STATUTE", "PERSON", "ORG"]
    },
    "general": {
        "model": "dslim/bert-base-NER",
        "entities": ["PER", "ORG", "LOC", "MISC"]
    }
}
