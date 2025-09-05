from typing import List
from chardet.universaldetector import UniversalDetector
import os
import torch
import shutil

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema import BaseRetriever
from sentence_transformers import SentenceTransformer, util

from FlagEmbedding import FlagReranker

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from configs import BaseConfig
from .utils import LineBreakTextSplitter, SingleFileSplitter

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

def get_embed_model(cfg: BaseConfig,
                    device: str = 'cpu',
                    retrival_database_batch_size: int = 256) -> OpenAIEmbeddings:
    """
    Get the embedding model based on the configuration.
    """
    if cfg.embedding.provider == 'openai':
        embed_model = OpenAIEmbeddings()
    elif cfg.embedding.provider == 'hf':
        try:
            embed_model = HuggingFaceEmbeddings(
                model_name=cfg.embedding.model_dir,
                model_kwargs={'device': device},
                encode_kwargs={'device': device, 'batch_size': retrival_database_batch_size},
            )
        except cfg.embedding.model_dir:
            raise Exception(f"Encoder {cfg.embedding.model_dir} not found, please check.")
    
    return embed_model

def build_prepare(cfg: BaseConfig, retrival_database_batch_size: int, device: str):
    # load raw data
    docs =[]
    for data_path in cfg.datastorage.raw_data_dir:
        docs.extend(load_files2docs(data_path))

    # get data chunk
    chunk_docs = chunk_documents(docs, cfg)

    # get embedding model
    embed_model = get_embed_model(cfg, device=device, retrival_database_batch_size=retrival_database_batch_size)

    return chunk_docs, embed_model

def chroma_database(cfg: BaseConfig, retrieval_store_path: str, retrieval_name: str, retrival_database_batch_size: int, device: str):
    # get retrieval data
    if os.path.exists(retrieval_store_path) and os.listdir(retrieval_store_path):
        print(f'loading {cfg.datastorage.tool} database of {retrieval_name} using {cfg.embedding.model_name}')
        embed_model = get_embed_model(cfg, device=device, retrival_database_batch_size=retrival_database_batch_size)
        retrieval_database = Chroma(
            embedding_function=embed_model,
            persist_directory=retrieval_store_path
        )
    else:
        print(f'generating {cfg.datastorage.tool} database of {retrieval_name} using {cfg.embedding.model_name}')
        chunk_docs, embed_model = build_prepare(cfg, retrival_database_batch_size, device)
        retrieval_database = Chroma.from_documents(
            documents=chunk_docs,
            embedding=embed_model,
            persist_directory=retrieval_store_path
        )

    return retrieval_database


def get_retriever(cfg: BaseConfig, force_rebuild: bool = False, retrival_database_batch_size: int = 512, with_database: bool = False ,device = 'cpu'):
    '''
    Get the data storage for the retrieval system, build if not constructed before or set force_rebuild True
    '''

    # get name and path
    retrieval_name, retrieval_store_path = get_retrieval_info(cfg)

    # if force rebiuld, clean old data
    if force_rebuild and os.path.exists(retrieval_store_path):
        print(f'force rebuild {cfg.datastorage.tool} database of {retrieval_name} using {cfg.embedding.model_name}')
        shutil.rmtree(retrieval_store_path)
        print("clean finished")
    
    if cfg.datastorage.tool == 'chroma':
        retrieval_database = chroma_database(cfg, retrieval_store_path, retrieval_name, retrival_database_batch_size, device)
    else:
        raise Exception(f"Datastore {cfg.datastorage.tool} not found, please check.")

    if cfg.retrieval.method == 'similarity_score_threshold':
        retriever: BaseRetriever = retrieval_database.as_retriever(
                search_type = cfg.retrieval.method,
                search_kwargs={"k": cfg.retrieval.params.get("k", 4),
                            'score_threshold': cfg.retrieval.params.get("score_threshold", 0.75)}  # get k, default 4
            )
        print(f"Retriever of {cfg.retrieval.method} is ready.")
    elif cfg.retrieval.method == 'mmr':
        retriever: BaseRetriever = retrieval_database.as_retriever(
                search_type = cfg.retrieval.method,
                search_kwargs={"k": cfg.retrieval.params.get("k", 4),
                            'fetch_k': cfg.retrieval.params.get("fetch_k", 8)}  # get k, default 4
            )
        print(f"Retriever of {cfg.retrieval.method} is ready.")

    print(f"Retriever of {cfg.datastorage.tool} is ready.")

    if with_database:
        return retriever, retrieval_database
    else:
        return retriever


def get_retrieved_contexts(cfg: BaseConfig, query: List[str], retriever: BaseRetriever, device: str = 'cpu') -> List[str]:
    '''
    Get the retrieved context from the retriever based on the query.
    using the as_retriever() interface.
    '''
    context=[]

    if cfg.retrieval.rerank:
        # rerank the documents based on similarity score
        reranker = FlagReranker(cfg.retrieval.rerank, devices=device, use_fp16=True)


    # docs = retriever.batch(query)
    for q in query:
        docs = retriever.invoke(q)

        if cfg.retrieval.rerank:
            pairs = [(q, con.page_content) for con in docs]
            scores = reranker.compute_score(pairs)
            reranked_docs = [doc for doc, score in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]

        context.append(cfg.retrieval.adhesive.join([doc.page_content for doc in reranked_docs]))
        
    return context