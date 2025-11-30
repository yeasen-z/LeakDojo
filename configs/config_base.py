from dataclasses import dataclass, field
from typing import Dict, Any, List
import datetime
import hashlib
import os
import re

class VRConfig:
    def __init__(self, dict_config: Dict = None):
        self.data = {
            "force_rebuild": False,
            "datastorage_tool": "chroma",
            "data_dir_list": ["./data/fiqa/corpus.jsonl"],
            "description": {
                "name": "fiqa",
                "type": "Finance",  # 映射英文类别
                "intro": "a financial sentiment analysis benchmark derived from real-world sources such as StockTwits posts and financial news headlines.it enables models to understand market sentiment and investor opinions in financial contexts."
            }
        }
        self.tool_llm = {
            "model": "./Models/Qwen3-14B",
            "base_url": "http://localhost:22999/v1",
            "api_key": "EMPTY",
            "temperature": 0.7,
            "top_p": 0.8
        }
        self.retrieval = {
            "method": "mmr",
            "top_k": 15,
            "fetch_k": 60,
            "score_threshold": 0.75,
            "top_n": 10,
            "embed": {
                "provider": "hf",
                "model_name": "bge-large-en-v1.5",
                "model_dir": "./Models/BAAI/bge-large-en-v1.5",
                "retrival_database_batch_size": 1024
            }
            # "embed": {
            #     "provider": "openai",
            #     "model_name": "bge-large-en-v1.5",
            #     "api_key": "YOUR_API_KEY",
            #     "retrival_database_batch_size": 256
            # }
        }
        self.reranker = {
            "provider": "hf",
            "model": "./Models/BAAI/bge-reranker-large",
            "api_key": None
        }
        self.extractor = {
            "provider": "hf",
            "model": "./Models/BAAI/bge-large-en-v1.5",
            "api_key": None
            # "provider": "openai",
            # "model": "BAAI/bge-large-en-v1.5",
            # "api_key": "YOUR_API_KEY"
        }

        if dict_config:
            self.update_4m_dict(dict_config)
        # 动态生成 retrieval 信息

        retrieval_name, retrieval_store_path = self.get_retrieval_info()

        self.data.update({
            "retrieval_name": retrieval_name,
            "retrieval_store_path": retrieval_store_path,
            "wbtq_filepath": [os.path.join(i.replace("corpus.jsonl", ""), "queries.jsonl") for i in self.data["data_dir_list"]]
        })

    def get_retrieval_info(self):
        """
        Get retrieval information from the configuration.
        """
        retrieval_name = self.data["description"]["name"]

        retrieval_store_path = f"./retrieval_stores/{retrieval_name}/{self.retrieval['embed']['model_name']}/{self.data['datastorage_tool']}"
        return retrieval_name, retrieval_store_path
    
    def update_4m_dict(self, config: dict):
        self.data.update(config.get("data", {}))
        self.tool_llm.update(config.get("tool_llm", {}))
        self.retrieval.update(config.get("retrieval", {}))
        self.reranker.update(config.get("reranker", {}))
        self.extractor.update(config.get("extractor", {}))
    
    def generate_exp_path(self, llm_generator_name: str):
        """
        路径结构: ./results/{DATASET}/{LLM_MODEL}/{RAG_CONFIG_TAG}/
        """
        def sanitize_filename(name: str) -> str:
            # 替换所有非法字符： / \ : * ? " < > | 和 空格
            if name is None:
                return "None"
            # 将路径分隔符替换为下划线，移除其他非法字符
            # 使用下划线替换所有不安全字符
            return re.sub(r'[\\/:\*\?"<>\|\s\.]+', '_', name.strip())

        dataset = sanitize_filename(os.path.basename(self.data.get("description", {}).get("name", "UnknownDataset")))
        llm_name = sanitize_filename(os.path.basename(llm_generator_name))
        rag_config_parts = []
        retrieval_method = self.retrieval.get("method", "BM25")
        retrieval_tag = f"R_{sanitize_filename(retrieval_method)}"
        embed_name = self.retrieval.get("embed", {}).get("model_name")
        if embed_name and retrieval_method.lower() not in ['bm25', 'fid']:
            retrieval_tag = f"R__{sanitize_filename(embed_name)}" 
        retrieval_tag += f"_k{self.retrieval.get('top_k', 5)}"
        rag_config_parts.append(retrieval_tag)

        if self.reranker:
            reranker_mdl = self.reranker.get("model", "UnknownReranker")
            reranker_tag = f"RR__{sanitize_filename(os.path.basename(reranker_mdl))}_n{self.retrieval.get('top_n', 3)}"
            rag_config_parts.append(reranker_tag)
        else:
            rag_config_parts.append(f"NoRR_n{self.retrieval.get('top_n', 3)}") # 即使没有 Reranker，也记录最终的 Top N

        if self.extractor:
            extractor_mdl = self.extractor.get("model", "UnknownExtractor")
            extractor_tag = f"EX__{sanitize_filename(os.path.basename(extractor_mdl))}"
            rag_config_parts.append(extractor_tag)
        else:
            rag_config_parts.append("NoEX")
            
        rag_config_tag = "-".join(rag_config_parts)

        # --- 3. 最终路径结构 ---
        save_dir = os.path.join(
            "./results",
            dataset,
            llm_name,
            rag_config_tag,
            ""
        )
        print(f"[INFO] Generated save directory: {save_dir}")
        return save_dir

    def generate_exp_filename(self, args: Any, suf_route: str, ext: str = ".jsonl") -> str:
        """
        生成完整的、可稳定重现的实验结果文件名。
        Args:
            args: 包含 attack, rewriter, reranker, extractor 等运行时参数的对象。
            ext: 文件扩展名 (默认为 .jsonl)。
        """
        
        def sanitize_filename(name: str) -> str:
            # 替换所有非法字符： / \ : * ? " < > | 和 空格, 且替换 . 为 _
            if name is None:
                return "None"
            return re.sub(r'[\\/:\*\?"<>\|\s\.]+', '_', name.strip())

        attack_method = sanitize_filename(getattr(args, "attack", "no_attack").upper()) 

        filename = (
            f"{attack_method}_"
            f"RW-{int(getattr(args, 'rewriter', False))}_" # 1/0 表示是否启用
            f"RR-{int(getattr(args, 'reranker', False))}_"
            f"EX-{int(getattr(args, 'extractor', False))}_"
            f"IF-{int(getattr(args, 'intent_filter', False))}_"
            f"OF-{int(getattr(args, 'output_filter', False))}_"
            f"{sanitize_filename(suf_route)}"
            f"{ext}"
        )
        
        print(f"[INFO] Generated save path: {filename}")
        return filename