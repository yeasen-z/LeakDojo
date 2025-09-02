from dataclasses import dataclass

@dataclass
class DataConfig:
    data_dir: str = "./data"
    vectorstore_dir: str = "./stores_vector/chroma"

@dataclass
class EmbeddingConfig:
    provider: str = "hf"  # "hf" | "openai"
    model: str = "models/embeddings/all-MiniLM-L6-v2"

@dataclass
class LLMConfig:
    provider: str = "hf"  # "openai" | "hf" | "local"
    model_name: str = "llama-2-7b-chat-hf"
    temperature: float = 0.0

@dataclass
class RetrievalConfig:
    k: int = 4
    score_threshold: float = 0.0