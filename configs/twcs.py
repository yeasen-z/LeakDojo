from configs.config_base import *

twcs = VRConfig(
    {
        "data": {
            "force_rebuild": False,
            "datastorage_tool": "chroma",
            "data_dir_list": ["./data/twcs/threads.jsonl"],
            "description": {
                "name": "twcs",
                "type": "Social Media/Chat",  # 映射英文类别
                "intro": "a social media corpus containing threads and related chat texts, useful for natural language processing tasks in the social media domain."
            }
        },
        "tool_llm": {
            # "model": "./Models/Qwen2.5-14B-Instruct",
            "model": "./Models/Qwen3-14B",
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
