from typing import List, Dict
import os
from langchain.schema import Document
from configs import VectorBaseConfig
import json

def get_retrieval_info(cfg: VectorBaseConfig):
    """
    Get retrieval information from the configuration.
    """
    retrieval_name = '_'.join(cfg.datastorage.raw_data_dir)
    if len(cfg.datastorage.raw_data_dir) != 1:
        retrieval_name = 'mix_' + retrieval_name

    retrieval_store_path = f"./retrieval_stores/{retrieval_name}/{cfg.embedding.model_name}/{cfg.datastorage.tool}"
    return retrieval_name, retrieval_store_path

def load_corpus(paths: List[str]):
    """加载 BEIR corpus.jsonl"""
    corpus = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc["_id"]] = {
                    "title": doc.get("title", ""),
                    "text": doc.get("text", "")
                }
    return corpus

def corpus_to_documents(corpus: Dict, cfg: VectorBaseConfig):
    """
    将 BEIR corpus 转换为 LangChain Documents，并自动分chunk
    """
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=cfg.chunk.params.get("chunk_size", 500),
    #     chunk_overlap=cfg.chunk.params.get("chunk_overlap", 50),
    #     length_function=len,
    # )

    documents = []
    for doc_id, doc in corpus.items():
        full_text = (doc["title"] + "\n" + doc["text"]).strip()
        # chunks = text_splitter.split_text(full_text)
        documents.append(
            Document(
                page_content=full_text,
                metadata={
                    "doc_id": doc_id,
                    "title": doc["title"],
                }
            )
        )
        # for i, chunk in enumerate(chunks):
        #     documents.append(
        #         Document(
        #             page_content=chunk,
        #             metadata={
        #                 "doc_id": doc_id,
        #                 "title": doc["title"],
        #                 "chunk_id": i,
        #             }
        #         )
        #     )
    return documents


def get_data_chunks(cfg: VectorBaseConfig):
    # load data in beir format
    data_path = []
    for raw_data_dir in cfg.datastorage.raw_data_dir:
        data_path.append(os.path.join(raw_data_dir, "corpus.jsonl"))
        
    data = load_corpus(data_path)

    # get data chunk
    chunk_docs = corpus_to_documents(data, cfg)

    return chunk_docs


def get_queries(cfg: VectorBaseConfig, suffix: str, query_file: str = "queries.jsonl"):
    # load queries in beir format
    data_paths = []  # add multiple dataset
    for path in cfg.datastorage.raw_data_dir:
        data_paths.append(os.path.join(path, query_file))
    queries = []
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                queries.append(doc["text"] + suffix)
    return queries

def get_queries_id(cfg: VectorBaseConfig):
    # load queries in beir format
    data_paths = []  # add multiple dataset
    for path in cfg.datastorage.raw_data_dir:
        data_paths.append(os.path.join(path, "queries.jsonl"))
    queries_id = []
    for data_path in data_paths:
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                queries_id.append(doc["_id"])
    return queries_id

def get_llm_output_file(cfg: VectorBaseConfig):
    model_name = os.path.basename(cfg.llm.model_name)
    return f"outputs-{model_name}-{cfg.llm.temperature}-{cfg.llm.max_seq_len}-{cfg.llm.max_gen_len}.json"

