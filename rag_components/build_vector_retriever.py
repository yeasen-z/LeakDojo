from typing import List
import os
import torch
import shutil

from langchain.schema import BaseRetriever

from FlagEmbedding import FlagReranker

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS

from configs import VectorBaseConfig
from .rag_utils import get_retrieval_info, get_data_chunks


def vector_embed_model(cfg: VectorBaseConfig,
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
                encode_kwargs={'device': device, 'batch_size': retrival_database_batch_size,"normalize_embeddings": True}
                )
        except cfg.embedding.model_dir:
            raise Exception(f"Encoder {cfg.embedding.model_dir} not found, please check.")
    
    return embed_model

def vector_chroma_database(cfg: VectorBaseConfig, retrieval_store_path: str, retrieval_name: str, retrival_database_batch_size: int, device: str):
    # get retrieval data
    if os.path.exists(retrieval_store_path) and os.listdir(retrieval_store_path):
        print(f'loading {cfg.datastorage.tool} database of {retrieval_name} using {cfg.embedding.model_name}')
        embed_model = vector_embed_model(cfg, device=device, retrival_database_batch_size=retrival_database_batch_size)
        retrieval_database = Chroma(
            embedding_function=embed_model,
            persist_directory=retrieval_store_path
        )
    else:
        print(f'generating {cfg.datastorage.tool} database of {retrieval_name} using {cfg.embedding.model_name}')
        chunk_docs = get_data_chunks(cfg)
        embed_model = vector_embed_model(cfg, device=device, retrival_database_batch_size=retrival_database_batch_size)
        retrieval_database = Chroma.from_documents(
            documents=chunk_docs,
            embedding=embed_model,
            persist_directory=retrieval_store_path
        )

    return retrieval_database


def vector_retriever(cfg: VectorBaseConfig, force_rebuild: bool = False, retrival_database_batch_size: int = 512, with_database: bool = False , device = 'cpu'):
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
    
    if 'chroma' in cfg.datastorage.tool:
        retrieval_database = vector_chroma_database(cfg, retrieval_store_path, retrieval_name, retrival_database_batch_size, device)
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


def vector_retrieved_contexts(cfg: VectorBaseConfig, query: List[str], retriever: BaseRetriever, join_adhesive: bool = False, device: str = 'cpu') -> List[str]:
    '''
    Get the retrieved context from the retriever based on the query.
    using the as_retriever() interface.
    return the united context and the sources.
        - context: List[str], the united context for each query
        - doc_ids: List[List[str]], the sources for each query, which is a list of list of sources
    '''
    context=[]
    doc_ids=[]

    if cfg.retrieval.rerank:
        # rerank the documents based on similarity score
        reranker = FlagReranker(cfg.retrieval.rerank, devices=device, use_fp16=True)


    # docs = retriever.batch(query)
    for q in query:
        docs = retriever.invoke(q)

        if cfg.retrieval.rerank:
            pairs = [(q, con.page_content) for con in docs]
            if pairs and len(pairs) > 0:
                scores = reranker.compute_score(pairs)
                reranked_docs = [doc for doc, score in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]
            else:
                reranked_docs = docs
                print("Warning: No documents retrieved for the query.", q)

        if join_adhesive:
            context.append(cfg.retrieval.adhesive.join([doc.page_content for doc in reranked_docs]))
        else:
            context.append([doc.page_content for doc in reranked_docs])

        doc_ids.append([doc.metadata.get("doc_id", "unknown") for doc in reranked_docs])

    return context, doc_ids