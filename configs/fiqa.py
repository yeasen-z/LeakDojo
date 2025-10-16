from configs.config_base import *

database_desc = {
    "name": "fiqa",
    "type": "Finance",  # 映射英文类别
    "intro": "a financial sentiment analysis benchmark derived from real-world sources such as StockTwits posts and financial news headlines.it enables models to understand market sentiment and investor opinions in financial contexts."
}


@dataclass
class Zeng24fiqa(VectorBaseConfig):
    # 只覆盖需要修改的部分
    datastorage: vDataStorageConfig = vDataStorageConfig(
        data_name="fiqa",
        raw_data_dir = ["./data/fiqa"],
        tool = "vector-chroma"
    )
    embedding: vEmbeddingConfig = vEmbeddingConfig(
        provider="hf",
        model_name = "bge-large-en-v1.5",
        model_dir = "./Models/BAAI-bge-large-en-v1.5"
    )
    retrieval: vRetrievalConfig = vRetrievalConfig(
        method="mmr",
        # # method="BM25",
        # rerank = 'BAAI/bge-reranker-large',
        # # rerank = "./Models/ms-marco-TinyBERT-L2-v2",
        # # rerank = "./Models/ms-marco-MiniLM-L6-v2",
        # # rerank=None,
        # adhesive = "\n\n",
        params={
            "k": 15,
            "n": 10,
            "fetch_k": 40
        },

        # method="similarity_score_threshold",

        rerank = 'BAAI/bge-reranker-large',
        adhesive = "\n\n",
        # params={
        #     "k": 15,
        #     "n": 10,
        #     "score_threshold": 0.5
        # }
    )