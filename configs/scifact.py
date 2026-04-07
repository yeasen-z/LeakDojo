from configs.config_base import *

scifact = VRConfig(
    {
        "data": {
            "force_rebuild": False,
            "datastorage_tool": "chroma",
            "data_dir_list": ["./data/scifact/corpus.jsonl"],
            "description": {
                "name": "scifact",
                "type": "Academic/Research",  # 映射英文类别
                "intro": "a dataset of expert-written scientific claims paired with evidence-containing abstracts, and annotated with labels and rationales."
            }
        },
        "tool_llm": {
            "model": "gpt-4.1-nano",
            "base_url": "https://aihubmix.com/v1",
            "api_key": "YOUR_API_KEY_HERE",
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
