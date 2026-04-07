from typing import List, Dict
import os
from langchain.schema import Document
import json
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

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


def corpus_to_documents_params(corpus: Dict):
    """
    将 BEIR corpus 转换为 LangChain Documents，并自动分chunk
    """
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
    return documents

def get_data_chunks_by_params(data_paths: List[str]):
    # load data in beir format
    data_path = []
    for raw_data_dir in data_paths:
        data_path.append(os.path.join(raw_data_dir))
        
    data = load_corpus(data_path)

    # get data chunk
    chunk_docs = corpus_to_documents_params(data)

    return chunk_docs

def load_save_helper(file_path: str) -> Dict:

    # 假设文件名为 data.json
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def get_embed_model(embed_provider, embed_model_dir, retrival_database_batch_size=128, device="cuda:1"):
    if embed_provider == 'openai':
        embed_model = OpenAIEmbeddings()
    elif embed_provider == 'hf':
        try:
            embed_model = HuggingFaceEmbeddings(
                model_name=embed_model_dir,
                model_kwargs={'device': device},
                encode_kwargs={'device': device, 'batch_size': retrival_database_batch_size, "normalize_embeddings": True}
                )
        except ValueError:
            raise Exception(f"Encoder {embed_model_dir} not found, please check.")
        print(f"Loaded embedding model from {embed_model_dir}")
    return embed_model