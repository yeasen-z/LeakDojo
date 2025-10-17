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

        save_dir = f"./exp/{dataset}-{store}/{embed}-{llm}/{retrieved_method}-{k}-{reranker_mdl}-{n}/{summarizer_mdl}/"
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
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_tag = hashlib.md5(f"{attack}_{time_str}".encode()).hexdigest()[:6]

        # === 文件名组合 ===
        filename = (
            f"rewr-{rewriter}_rerank-{reranker}_sum-{summarizer}_"
            f"{attack}_{time_str}_{unique_tag}{ext}"
        )

        return filename

# @dataclass
# class vDataStorageConfig:
#     data_name: str = "fiqa"
#     raw_data_dir: List[str] = field(default_factory=List)
#     tool: str = "vector-chroma"  # # "vector-chroma"

# @dataclass
# class vEmbeddingConfig:
#     provider: str = "hf"  # "hf" | "openai"
#     model_name: str = "bge-large-en-v1.5",
#     model_dir: str = "BAAI/bge-large-en-v1.5"

# @dataclass
# class vRetrievalConfig:
#     method: str = "similarity_score_threshold"  # "similarity_score_threshold" | "mmr"
#     rerank: str = 'BAAI/bge-reranker-large'
#     adhesive: str = "\n\n"
#     params: Dict[str, Any] = field(default_factory=lambda: {
#         "k": 10,
#         "n": 3,
#         "score_threshold": 0.75
#     })


# @dataclass
# class vExpConfig:
#     output_dir: str = "./exp/demo/"


# @dataclass
# class VectorBaseConfig:
#     datastorage: vDataStorageConfig = vDataStorageConfig(
#         data_name = "chatdoctor",
#         raw_data_dir = ["./data/chatdoctor"],
#         tool = "vector-chroma"
#     )
#     embedding: vEmbeddingConfig = vEmbeddingConfig(
#         provider="hf",
#         model_name = "all-MiniLM-L6-v2",
#         model_dir = "all-MiniLM-L6-v2"
#     )
#     retrieval: vRetrievalConfig = vRetrievalConfig(
#         method = "mmr",
#         rerank = None,
#         adhesive = "\n\n",
#         params = {
#             "k": 10,
#             "n": 3,
#             "fetch_k": 40
#         }
#     )

#     @property
#     def expconfig(self) -> vExpConfig:
#         """动态生成实验输出目录"""

#         def sanitize_filename(name: str) -> str:
#             """
#             将字符串中不适合用于文件路径的字符替换为下划线。
#             """
#             # 替换所有非法字符： / \ : * ? " < > | 和 空格
#             return re.sub(r'[\\/:\*\?"<>\|\s]+', '_', name.strip())
        
#         # dataset = os.path.basename(self.datastorage.data_name)  # chatdoctor
#         dataset = sanitize_filename(os.path.basename(self.datastorage.data_name))
#         # store = self.datastorage.tool                           # vector-chroma
#         store = sanitize_filename(self.datastorage.tool)
#         # embed = self.embedding.model_name.replace(".","_")                       # bge-large-en-v1.5  -> bge-large-en-v1_5
#         embed = sanitize_filename(self.embedding.model_name.replace(".", "_"))
#         # llm = os.path.basename(self.llm.model_name).replace(".","_")             # Llama-2-13b-chat-hf
#         llm = sanitize_filename(os.path.basename(self.llm.model_name).replace(".", "_"))
#         k = self.retrieval.params.get("k", 2)                                    # 2
#         # retrieved_method = self.retrieval.method
#         retrieved_method = sanitize_filename(self.retrieval.method)
#         # reranker = self.retrieval.rerank if self.retrieval.rerank else "no-rerank"
#         reranker = sanitize_filename(self.retrieval.rerank if self.retrieval.rerank else "no-rerank")

#         save_dir = f"./exp/{dataset}/{store}/{embed}-{llm}/{retrieved_method}-{k}-{reranker}/"
#         return vExpConfig(output_dir=save_dir)