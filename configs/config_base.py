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
            "data_dir_list": ["./data/fiqa"],
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
                "model_dir": "./Models/BAAI-bge-large-en-v1.5",
                "retrival_database_batch_size": 256
            }
        }
        self.reranker = {
            "model": "BAAI/bge-reranker-large",
        }
        self.summarizer = {
            "provider": "hf",
            "model": "./Models/BAAI-bge-large-en-v1.5"
        }

        if dict_config:
            self.update_4m_dict(dict_config)
        # 动态生成 retrieval 信息

        retrieval_name, retrieval_store_path = self.get_retrieval_info()

        self.data.update({
            "retrieval_name": retrieval_name,
            "retrieval_store_path": retrieval_store_path,
            "wbtq_filepath": [os.path.join(i, "queries.jsonl") for i in self.data["data_dir_list"]]
        })

    def get_retrieval_info(self):
        """
        Get retrieval information from the configuration.
        """
        retrieval_name = '_'.join(self.data["data_dir_list"])
        if len(self.data["data_dir_list"]) != 1:
            retrieval_name = 'mix_' + retrieval_name

        retrieval_store_path = f"./retrieval_stores/{retrieval_name}/{self.retrieval['embed']['model_name']}/{self.data['datastorage_tool']}"
        return retrieval_name, retrieval_store_path
    
    def update_4m_dict(self, config: dict):
        self.data.update(config.get("data", {}))
        self.tool_llm.update(config.get("tool_llm", {}))
        self.retrieval.update(config.get("retrieval", {}))
        self.reranker.update(config.get("reranker", {}))
        self.summarizer.update(config.get("summarizer", {}))

    def generate_expconfig(self, llm_model: str):
        def sanitize_filename(name: str) -> str:
            # 替换所有非法字符： / \ : * ? " < > | 和 空格
            return re.sub(r'[\\/:\*\?"<>\|\s]+', '_', name.strip())
        
        dataset = sanitize_filename(os.path.basename(self.data["description"]["name"]))
        store = sanitize_filename(self.data["datastorage_tool"])
        embed = sanitize_filename(self.retrieval["embed"]["model_name"].replace(".", "_"))
        llm = sanitize_filename(os.path.basename(llm_model).replace(".", "_"))
        k = self.retrieval["top_k"]
        n = self.retrieval["top_n"]
        retrieved_method = sanitize_filename(self.retrieval["method"])
        reranker_mdl = sanitize_filename(os.path.basename(self.reranker["model"]).replace(".", "_"))
        summarizer_mdl = sanitize_filename(os.path.basename(self.summarizer["model"]).replace(".", "_"))
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_tag = hashlib.md5(f"{time_str}".encode()).hexdigest()[:6]
        save_dir = f"./exp/{dataset}-{store}/{embed}-{llm}/{retrieved_method}-{k}-{reranker_mdl}-{n}/{summarizer_mdl}/{unique_tag}/"
        return save_dir

    def generate_expfilename(self, args=None, ext=".json"):
        """
        生成唯一的实验文件名（不包含路径）。
        文件名中会包含关键信息（dataset、attack、llm、retrieval参数、时间戳等）。
        """
        def sanitize(name: str) -> str:
            return re.sub(r'[\\/:\*\?"<>\|\s]+', '_', str(name).strip())

        # === 运行参数 ===
        attack = getattr(args, "attack", "noatk") if args else "noatk"
        rewriter = getattr(args, "rewriter", False)
        reranker = getattr(args, "reranker", False)
        summarizer = getattr(args, "summarizer", False)

        # === 唯一标识（时间戳 + 哈希） ===
        # time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # unique_tag = hashlib.md5(f"{attack}_{time_str}".encode()).hexdigest()[:6]

        # === 文件名组合 ===
        filename = (
            f"rewr-{rewriter}_rerank-{reranker}_sum-{summarizer}_"
            f"{attack}{ext}"
        )

        return filename