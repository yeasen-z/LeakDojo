from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Iterable
import os, shutil, torch
from FlagEmbedding import FlagReranker
from langchain.schema import BaseRetriever
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# from longchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from configs import VectorBaseConfig
from .utils import get_retrieval_info, get_data_chunks

from .interfaces import Retriever, Reranker


class VectorRetriever(Retriever):
    def __init__(self, cfg: VectorBaseConfig, device: str = 'cpu', force_rebuild: bool = False, retrival_database_batch_size: int = 256):
        self.cfg = cfg
        self.device = device
        self.force_rebuild = force_rebuild
        self.retrival_database_batch_size = retrival_database_batch_size

        # 准备相关信息，数据库名称以及保存地址
        retrieval_name, retrieval_store_path = get_retrieval_info(cfg)
        print(f"[INFO] Retrieval name: {retrieval_name}", f"Store path: {retrieval_store_path}")

        # 检查是否为BM25, 如果是，跳过向量数据库建立阶段，直接建立检索器
        if cfg.retrieval.method == 'BM25':
            self.database = None
            self.retriever = self._build_retriever()
            print(f"[INFO] BM25 Retriever for {retrieval_name} is ready!")
            return
        
        # 如果是向量数据库模型
        # 是否强制重建
        if self.force_rebuild and os.path.exists(retrieval_store_path):
            print(f"[INFO] Force rebuild {retrieval_name}")
            shutil.rmtree(retrieval_store_path)

        # 构建向量数据库
        if 'chroma' in cfg.datastorage.tool:
            self.database = self._build_chroma_database(retrieval_store_path, retrieval_name)
        else:
            raise Exception(f"Datastore {cfg.datastorage.tool} not supported")
        
        self.retriever = self._build_retriever()
        print(f"[INFO] Retriever for {retrieval_name} is ready!")

    def _embed_model(self):
        if self.cfg.embedding.provider == 'openai':
            embed_model = OpenAIEmbeddings()
        elif self.cfg.embedding.provider == 'hf':
            try:
                embed_model = HuggingFaceEmbeddings(
                    model_name=self.cfg.embedding.model_dir,
                    model_kwargs={'device': self.device},
                    encode_kwargs={'device': self.device, 'batch_size': self.retrival_database_batch_size,"normalize_embeddings": True}
                    )
            except self.cfg.embedding.model_dir:
                raise Exception(f"Encoder {self.cfg.embedding.model_dir} not found, please check.")
        return embed_model

    def _build_chroma_database(self, retrieval_store_path: str, retrieval_name: str):
        embed_model = self._embed_model()
        if os.path.exists(retrieval_store_path) and os.listdir(retrieval_store_path):
            # existing db
            print(f"[INFO] Loading existing Chroma DB: {retrieval_name}")
            db = Chroma(embedding_function=embed_model,
                        persist_directory=retrieval_store_path)
        else:
            # new db
            print(f"[INFO] Building new Chroma DB: {retrieval_name}")
            chunk_docs = get_data_chunks(self.cfg)
            db = Chroma.from_documents(
                documents=chunk_docs,
                embedding=embed_model,
                persist_directory=retrieval_store_path,
                collection_metadata={"hnsw:space": "cosine"}
            )
        return db
    
    def _build_retriever(self) -> BaseRetriever:
        cfg = self.cfg
        if cfg.retrieval.method == 'similarity_score_threshold':
            retriever: BaseRetriever = self.database.as_retriever(
                    search_type = cfg.retrieval.method,
                    search_kwargs={"k": cfg.retrieval.params.get("k", 4),
                                'score_threshold': cfg.retrieval.params.get("score_threshold", 0.75)}  # get k, default 4
                )
            print(f"Retriever of {cfg.retrieval.method} is ready.")
        elif cfg.retrieval.method == 'mmr':
            retriever: BaseRetriever = self.database.as_retriever(
                    search_type = cfg.retrieval.method,
                    search_kwargs={"k": cfg.retrieval.params.get("k", 4),
                                'fetch_k': cfg.retrieval.params.get("fetch_k", 8)}  # get k, default 4
                )
            print(f"Retriever of {cfg.retrieval.method} is ready.")
        elif cfg.retrieval.method == 'BM25':
            docs = get_data_chunks(cfg)
            retriever: BaseRetriever = BM25Retriever.from_documents(docs, k=cfg.retrieval.params.get("k", 4))

        print(f"Retriever of {cfg.datastorage.tool} is ready.")
        return retriever

    def _ensure_list_of_str(self, x: Union[str, Iterable[str]]) -> List[str]:
        """Utility: 把单个 str 或可迭代[str] 统一成 List[str]."""
        if isinstance(x, str):
            return [x]
        return list(x)
    
    def _unique_docs_preserve_order(self, docs: List) -> List:
        """
        按 page_content 去重，保持原有顺序。
        如果内容完全一致（忽略首尾空格），则只保留第一个。
        """
        seen = set()
        unique = []

        for doc in docs:
            content = getattr(doc, "page_content", None)
            if content is None:
                # 没有内容的文档，用唯一对象 id 保证不被误去重
                key = f"none_{id(doc)}"
            else:
                # 使用 strip 去除首尾空格影响
                key = content.strip()

            if key in seen:
                continue

            seen.add(key)
            unique.append(doc)

        return unique

    def retrieve(self, query: Union[List[str], List[List[str]]]) -> Tuple[List, List]:
        """输入多个查询，返回每个查询对应的多个检索结果"""
        cfg = self.cfg
        all_contexts, all_doc_ids = [], []
        for q in query:
            # 统一为改写列表（可能是单条字符串或列表）
            rewrites = self._ensure_list_of_str(q)

            # 收集来自每个 rewrite 的检索结果
            docs_aggregated = []
            for rw in rewrites:
                docs = self.retriever.invoke(rw)

                # 支持单个 doc 或 list 返回，统一为 list
                if docs is None:
                    docs = []
                elif not isinstance(docs, (list, tuple)):
                    docs = [docs]

                docs_aggregated.extend(docs)

            # 合并 + 去重（按 doc_id 或 id 或 page_content）
            docs_uniq = self._unique_docs_preserve_order(docs_aggregated)

            # docs = self.retriever.invoke(q)
            
            # # 如果不启用 reranker，则仅保留前 n 个
            # if cfg.retrieval.rerank == None:
            #     top_docs = docs[:cfg.retrieval.params.get("n", 3)]
            #     print(f"[INFO] Retrieved {len(docs)} docs, return top {len(top_docs)} docs without reranking.")
            # else:
            #     # 有独立的 reranker 类，这里不 rerank，外部调用时再做
            #     top_docs = docs

            # # 收集对应的 context
            # all_contexts.append([doc.page_content for doc in top_docs])

            # # 收集对应的 doc id
            # all_doc_ids.append([
            #     doc.metadata.get("doc_id", getattr(doc, "id", "unknown"))
            #     for doc in top_docs
            # ])

            # 如果不启用 reranker，则仅保留前 n 个
            if cfg.retrieval.rerank == None:
                top_docs = docs_uniq[:cfg.retrieval.params.get("n", 3)]
                print(f"[INFO] Retrieved {len(docs_uniq)} docs, return top {len(top_docs)} docs without reranking.")
            else:
                # 有独立的 reranker 类，这里不 rerank，外部调用时再做
                top_docs = docs_uniq

            # 收集对应的 context
            all_contexts.append([doc.page_content for doc in top_docs])

            # 收集对应的 doc id
            all_doc_ids.append([
                doc.metadata.get("doc_id", getattr(doc, "id", "unknown"))
                for doc in top_docs
            ])

        return all_contexts, all_doc_ids


class RerankerManager(Reranker):
    def __init__(self, cfg: VectorBaseConfig, device: str = 'cpu'):
        self.cfg = cfg
        self.device = device
        
        # 准备 reranker, 如果config没有，那么不应该调用reranker的生成，程序应当报错
        if cfg.retrieval.rerank:
            # rerank the documents based on similarity score
            self.reranker = FlagReranker(cfg.retrieval.rerank, devices=device, use_fp16=True)
            print(f"[INFO] Reranker {cfg.retrieval.rerank} is ready!")
        else:
            raise ValueError(
                "[ERROR] No reranker specified in config. Please set `cfg.retrieval.rerank` to a valid model name (e.g., 'BAAI/bge-reranker-large')."
            )

    def rerank(self, docs: List[List[str]], docs_id: List[List[str]], queries: List[str]) -> List[List[str]]:
        """
        输入:
            queries: 查询列表
            docs: 每个查询对应的文档内容列表 [["content1", "content2", ...], ...]
            docs_id: 每个查询对应的文档 ID 列表 [["id1", "id2", ...], ...]
        输出:
            reranked_docs: 每个查询对应的重排后文档内容列表
            reranked_doc_ids: 每个查询对应的重排后文档 ID 列表
        """

        all_reranked_docs = []
        all_reranked_doc_ids = []
        n = self.cfg.retrieval.params.get("n", 3)

        for query, doc_list, doc_id_list in zip(queries, docs, docs_id):
            if not doc_list:
                print(f"Warning: No documents to rerank for query: {query}")
                all_reranked_docs.append([])
                all_reranked_doc_ids.append([])
                continue

            # 生成 (query, doc_content) 对用于打分
            pairs = [(query, content) for content in doc_list]

            # 计算得分
            scores = self.reranker.compute_score(pairs)
            assert len(scores) == len(doc_list), "scores 数量与文档数量不匹配"

            # 按分数降序排序
            ranked = sorted(zip(doc_id_list, doc_list, scores), key=lambda x: x[2], reverse=True)

            # 分别提取 top-n 文档和对应 ID
            reranked_doc_ids = [doc_id for (doc_id, _, _) in ranked[:n]]
            reranked_docs = [content for (_, content, _) in ranked[:n]]

            # 收集结果
            all_reranked_docs.append(reranked_docs)
            all_reranked_doc_ids.append(reranked_doc_ids)

        return all_reranked_docs, all_reranked_doc_ids