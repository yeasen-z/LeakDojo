from configs.config_base import *

nfcorpus = VRConfig(
    {
        "data": {
            "force_rebuild": False,
            "datastorage_tool": "chroma",
            "data_dir_list": ["./data/nfcorpus/corpus.jsonl"],
            "description": {
                "name": "nfcorpus",
                "type": "Medical/Clinical",  # 映射英文类别
                "intro": "a medical corpus containing clinical notes and related medical texts, useful for natural language processing tasks in the healthcare domain."
            }
        },
        "tool_llm": {
            "model": "gpt-4.1-mini",
            "base_url": "https://aihubmix.com/v1",
            "api_key": "sk-XWaGp10Cjy2pZfttA8E538967f7f4dA7A463F584C17b63Bf",
            "reasoning": True,
            "temperature": 0.7,
            "top_p": 0.8
        },
        "retrieval": {
            "method": "mmr",
            "top_k": 10,
            "fetch_k": 40,
            "score_threshold": 0.75,
            "top_n": 5,
            "embed": {
                "provider": "hf",
                "model_name": "bge-large-en-v1.5",
                "model_dir": "./Models/BAAI/bge-large-en-v1.5",
                "retrival_database_batch_size": 256
            }
        },
        "reranker": {
            "provider": "hf",
            "model": "./Models/BAAI/bge-reranker-large",
            "api_key": None
        },
        "extractor": {
            "provider": "hf",
            "model": "./Models/BAAI/bge-large-en-v1.5",
            "api_key": None
        }
    }
)
