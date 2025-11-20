from typing import List, Tuple, Union, Iterable
import os, shutil, torch
from FlagEmbedding import FlagReranker
from langchain.schema import BaseRetriever
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from configs import VRConfig
import re
import torch.nn.functional as F
import textwrap
from .utils import get_data_chunks_by_params

from src.interfaces import Retriever, Reranker, Summarizer, LLMManager

class VectorRetriever(Retriever):
    def __init__(self, config: VRConfig, device: str = 'cpu'):

        self.config = config
        self.device = device

        print(f"[INFO] Retrieval name: {self.config.data['retrieval_name']}", f"Store path: {self.config.data['retrieval_store_path']}")

        # 检查是否为BM25, 如果是，跳过向量数据库建立阶段，直接建立检索器
        if self.config.retrieval['method'] == 'BM25':
            self.database = None
            self.retriever = self._build_retriever()
            print(f"[INFO] BM25 Retriever for {self.config.data['retrieval_name']} is ready!")
            return
        
        # 如果是向量数据库模型
        # 是否强制重建
        if self.config.data['force_rebuild'] and os.path.exists(self.config.data['retrieval_store_path']):
            print(f"[INFO] Force rebuild {self.config.data['retrieval_name']}")
            shutil.rmtree(self.config.data['retrieval_store_path'])

        # 构建向量数据库
        if 'chroma' in self.config.data['datastorage_tool']:
            self.database = self._build_chroma_database(self.config.data['retrieval_store_path'], self.config.data['retrieval_name'])
        else:
            raise Exception(f"Datastore {self.config.data['datastorage_tool']} not supported")

        self.retriever = self._build_retriever()
        print(f"[INFO] Retriever for {self.config.data['retrieval_name']} is ready!")

    def _embed_model(self):
        if self.config.retrieval["embed"]['provider'] == 'openai':
            embed_model = OpenAIEmbeddings(
                model=self.config.retrieval["embed"].get('model_name', "BAAI/bge-large-en-v1.5"),
                api_key=self.config.retrieval["embed"].get('api_key', None),
                base_url="https://aihubmix.com/v1"
            )
        elif self.config.retrieval["embed"]['provider'] == 'hf':
            try:
                embed_model = HuggingFaceEmbeddings(
                    model_name=self.config.retrieval["embed"]['model_dir'],
                    model_kwargs={'device': self.device},
                    encode_kwargs={'device': self.device, 'batch_size': self.config.retrieval["embed"]["retrival_database_batch_size"], "normalize_embeddings": True}
                    )
            except self.config.retrieval["embed"]['model_dir']:
                raise Exception(f"Encoder {self.config.retrieval['embed']['model_dir']} not found, please check.")
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
            chunk_docs = get_data_chunks_by_params(self.config.data['data_dir_list'])
            db = Chroma.from_documents(
                documents=chunk_docs,
                embedding=embed_model,
                persist_directory=retrieval_store_path,
                collection_metadata={"hnsw:space": "cosine"}
            )
        return db
    
    def _build_retriever(self) -> BaseRetriever:
        if self.config.retrieval['method'] == 'similarity_score_threshold':
            retriever: BaseRetriever = self.database.as_retriever(
                    search_type = 'similarity_score_threshold',
                    search_kwargs={"k": self.config.retrieval["top_k"],
                                'score_threshold': self.config.retrieval['score_threshold']}  # get k, default 4
                )
            print(f"Retriever of {self.config.retrieval['method']} is ready.")
        elif self.config.retrieval['method'] == 'mmr':
            retriever: BaseRetriever = self.database.as_retriever(
                    search_type = 'mmr',
                    search_kwargs={"k": self.config.retrieval['top_k'],
                                'fetch_k': self.config.retrieval['fetch_k']}  # get k, default 4
                )
            print(f"Retriever of {self.config.retrieval['method']} is ready.")
        elif self.config.retrieval['method'] == 'BM25':
            docs = get_data_chunks_by_params(self.config.data['data_dir_list'])
            retriever: BaseRetriever = BM25Retriever.from_documents(docs, k=self.config.retrieval['top_k'])

        print(f"Retriever of {self.config.data['datastorage_tool']} is ready.")
        return retriever

    def _ensure_list_of_str(self, x: Union[str, Iterable[str]]) -> List[str]:
        """Utility: 把单个 str 或可迭代[str] 统一成 List[str]."""
        if x is None:
            return []

        if isinstance(x, str):
            x = x.strip()
            return [x] if x else []

        # 如果是可迭代类型，则过滤掉 None 和空字符串
        cleaned = []
        for item in x:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    cleaned.append(s)
        return cleaned
    
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
        """
            输入多个查询，返回每个查询对应的多个检索结果
            为了明确格式输入，实际上处理的为 List[List[str]]，即每个查询可能是单条字符串或多条改写
            但是如果没有rewriter，同样支持 List[str] 作为输入，不过仍会被处理为 List[List[str]] 的格式
            Args:
                query: List[str] 或 List[List[str]]，每个查询可能是单条字符串或多条改写
            Returns:
                all_contexts: List[List[str]]，每组查询对应的多个检索结果
                all_doc_ids: List[List[str]]，每组查询对应的多个检索结果
        """
        all_contexts, all_doc_ids = [], []
        for q in query:
            # 统一为改写列表（可能是单条字符串或列表）
            rewrites = self._ensure_list_of_str(q)

            # 收集来自每个 rewrite 的检索结果
            docs_aggregated = []
            for rw in rewrites:
                docs = self.retriever.invoke(rw)

                if docs is None:
                    docs = []
                elif not isinstance(docs, (list, tuple)):
                    docs = [docs]

                docs_aggregated.extend(docs)

            # 合并 + 去重（按 doc_id 或 id 或 page_content）
            top_docs = self._unique_docs_preserve_order(docs_aggregated)

            # 收集对应的 context
            all_contexts.append([doc.page_content for doc in top_docs])

            # 收集对应的 doc id
            all_doc_ids.append([
                doc.metadata.get("doc_id", getattr(doc, "id", "unknown"))
                for doc in top_docs
            ])

        return all_contexts, all_doc_ids

class RerankerManager(Reranker):
    def __init__(self, reranker_config: dict, top_n: int = 10, device: str = 'cpu'):
        self.device = device
        self.reranker_model = reranker_config.get("model", None)
        self.reranker_provider = reranker_config.get("provider", "hf")
        self.reranker_api_key = reranker_config.get("api_key", None)
        self.top_n = top_n

        # 准备 reranker, 如果config没有，那么不应该调用reranker的生成，程序应当报错
        if self.reranker_model and self.reranker_provider == 'hf':
            # rerank the documents based on similarity score
            self.reranker = FlagReranker(self.reranker_model, devices=device, use_fp16=True)
            print(f"[INFO] Reranker {self.reranker_model} is ready!")
        elif self.reranker_provider == 'openai':
            raise NotImplementedError("OpenAI reranker is not implemented yet.")
        else:
            raise ValueError(
                "[ERROR] No reranker specified in config. Please set `reranker_model` to a valid model name (e.g., 'BAAI/bge-reranker-large')."
            )

    def rerank(self, docs: List[List[str]], docs_id: List[List[str]], queries: List[List[str]]) -> List[List[str]]:
        """
        输入:
            docs: 每个查询对应的文档内容列表 [["content1", "content2", ...], ...]
            docs_id: 每个查询对应的文档 ID 列表 [["id1", "id2", ...], ...]
            queries: 查询列表，同样是列表的列表 [["query1"],["query2"],...]
        输出:
            reranked_docs: 每个查询对应的重排后文档内容列表
            reranked_doc_ids: 每个查询对应的重排后文档 ID 列表
        """

        all_reranked_docs = []
        all_reranked_doc_ids = []
        n = self.top_n

        for query, doc_list, doc_id_list in zip(queries, docs, docs_id):
            if not doc_list:
                print(f"Warning: No documents to rerank for query: {query[0] if query else 'N/A'}")
                all_reranked_docs.append([])
                all_reranked_doc_ids.append([])
                continue

            # 生成 (query, doc_content) 对用于打分
            pairs = [(query[0], content) for content in doc_list]

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

class LLMHybridSummarization(Summarizer):
    """ Hybrid抽取式压缩：分句 + 短句合并 + Embedding筛选 + Query-aware压缩， 使用LLM"""
    def __init__(self,
                 llm: LLMManager,
                 sum_config: dict,
                 device: str = 'cpu', 
                 retain_ratio: float = 0.8, 
                 retain_floor: int = 10,
                 short_sent_threshold: int = 200,
                 embed_batch_size: int = 256
                 ):
        self.llm = llm
        self.embed_provider = sum_config.get("provider", "hf")
        self.embed_model = sum_config.get("model", "./Models/BAAI/bge-large-en-v1.5")
        self.embed_api_key = sum_config.get("api_key", None)
        self.retain_ratio = retain_ratio
        self.retain_floor = retain_floor
        self.short_sent_threshold = short_sent_threshold
        self.device = device
        self.embed_batch_size = embed_batch_size
        self.embed_model = self._embed_model()

    def _split_and_merge_sentences(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        
        sents = re.split(r'(?<=[。！？.!?])\s*', text)
        sents = [s.strip() for s in sents if s.strip()]

        if not sents:
            return []

        merged, buffer = [], ""
        for s in sents:
            if len(s.split()) < self.short_sent_threshold:
                buffer += " " + s
            else:
                if buffer:
                    merged.append(buffer.strip())
                    buffer = ""
                merged.append(s)
        if buffer:
            merged.append(buffer.strip())

        return merged

    def _embed_model(self):
        if self.embed_provider == 'openai':
            embed_model = OpenAIEmbeddings(
                model=self.embed_model,
                api_key=self.embed_api_key,
                base_url="https://aihubmix.com/v1"
            )
        elif self.embed_provider == 'hf':
            try:
                embed_model = HuggingFaceEmbeddings(
                    model_name=self.embed_model,
                    model_kwargs={'device': self.device},
                    encode_kwargs={'device': self.device, 'batch_size': self.embed_batch_size,"normalize_embeddings": True}
                    )
                print(f"[INFO] Summarizer embedding model {self.embed_model} loaded successfully.")
            except self.embed_model:
                raise Exception(f"Encoder {self.embed_model} not found, please check.")
        return embed_model
    
    def _embedding_filter(self, sentences: List[str]) -> List[str]:
        if not sentences:
            return []
        if len(sentences) <= 2:
            return sentences

        try:
            embs = torch.tensor(self.embed_model.embed_documents(sentences), device=self.device)
        except Exception as e:
            print(f"[WARN] Embedding computation failed: {e}")
            return sentences
        
        if embs.ndim != 2 or embs.size(0) == 0:
            return sentences

        doc_emb = embs.mean(dim=0, keepdim=True)
        sims = F.cosine_similarity(embs, doc_emb)
        topk = max(1, self.retain_floor, int(len(sentences) * self.retain_ratio))
        top_idx = sims.topk(topk).indices.tolist()
        return [sentences[i] for i in sorted(top_idx)]

    def _query_filter(self, sentences: List[str], query: str) -> List[str]:
        
        extracted = []
    
        for s in sentences:
            prompt = textwrap.dedent(f"""
                Extract only the text spans from the sentence that contain key information relevant to the question.
                Question: {query}
                Sentence: {s}

                Guidelines:
                - Copy text exactly from the sentence whenever possible; do not paraphrase unless absolutely necessary for clarity.
                - Return all numerical information (numbers, percentages, units, and dates) exactly as they appear.
                - Include factual and conceptual details that directly answer the question.
                - Omit unrelated or background information.
                - If no relevant information is found, return "None".
                - If multiple relevant parts exist, separate them with semicolons. Do NOT add explanations, commentary, or formatting.
            """)
            
            try:
                response, _ = self.llm.infer(prompt)
                info = response.strip()
                if info:
                    extracted.append(info)
            except Exception as e:
                print(f"[WARN] LLM failed on sentence: {s}\nError: {e}")
                # 可选择 fallback: 用原句或跳过
                # extracted.append(s)
        
        return extracted


    def summarize(self, documents: List[List[str]], queries: List[List[str]]) -> List[List[str]]:
        """
            对输入的文档列表进行抽取式压缩，返回压缩后的文档列表
            这里的格式为：
            documents: List[List[str]]，每个查询对应的文档内容列表 [["content1", "content2", ...], ...]
            queries: List[List[str]]，每个查询对应的查询内容列表 [["query1"],["query2"],...]
            返回: List[List[str]]，每个查询对应的压缩后文档内容列表 [["sum_content1","sum_content2", ...], ...]
        """

        if not documents:
            return []

        sentenses = []
        for doc_list in documents:
            sentenses.append([])
            for doc in doc_list:
                out = self._split_and_merge_sentences(doc)
                sentenses[-1].extend(out)

        embed_filtered = []
        query_filtered = []
        for sentense_list, query in zip(sentenses, queries):
            embed_filtered.append(self._embedding_filter(sentense_list))
            query_filtered.append(self._query_filter(embed_filtered[-1], query[0]))

        return query_filtered