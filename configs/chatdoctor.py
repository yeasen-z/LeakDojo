from configs.config_base import *

chatdoctor = VRConfig(
    {
        "data": {
            "force_rebuild": False,
            "datastorage_tool": "chroma",
            "data_dir_list": ["./data/chatdoctor/corpus.jsonl"],
            "description": {
                "name": "chatdoctor",
                "type": "Medical/Clinical",  # 映射英文类别
                "intro": "a medical question answering benchmark derived from real-world sources such as medical forums and healthcare articles. it enables models to understand and respond to medical inquiries accurately."
            }
        },
        "tool_llm": {
            "model": "gpt-4.1-mini",
            "base_url": "https://aihubmix.com/v1",
            "api_key": "sk-TCSKjHDLiEXpd2bv5845DfCb87F74cE3A776Be8757E2F310",
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
                "retrival_database_batch_size": 512
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
