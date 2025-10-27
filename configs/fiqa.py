from configs.config_base import *

fiqa = VRConfig(
    {
        "data": {
            "force_rebuild": False,
            "datastorage_tool": "chroma",
            "data_dir_list": ["./data/fiqa"],
            "description": {
                "name": "fiqa",
                "type": "Finance",  # 映射英文类别
                "intro": "a financial sentiment analysis benchmark derived from real-world sources such as StockTwits posts and financial news headlines.it enables models to understand market sentiment and investor opinions in financial contexts."
            }
        },
        "tool_llm": {
            "model": "./Models/Qwen2.5-14B-Instruct",
            "base_url": "http://localhost:22999/v1",
            "api_key": "EMPTY",
            "reasoning": True,
            "temperature": 0.7,
            "top_p": 0.8
        },
        "retrieval": {
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
        },
        "reranker": {
            "model": "BAAI/bge-reranker-large",
        },
        "summarizer": {
            "provider": "hf",
            "model": "./Models/BAAI-bge-large-en-v1.5"
        }
    }
)
