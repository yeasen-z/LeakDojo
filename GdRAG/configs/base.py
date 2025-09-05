from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class DataStorageConfig:
    raw_data_dir: List[str] = field(default_factory=List)
    tool: str = "chroma"  # "chroma" | "faiss"

@dataclass
class ChunkConfig:
    method: str = "recursive"  # recursive | by_two_line_breaks | by_single_file
    params: Dict[str, Any] = field(default_factory=lambda: {
        "chunk_size": 1000,
        "chunk_overlap": 200
    })

@dataclass
class EmbeddingConfig:
    provider: str = "hf"  # "hf" | "openai"
    model_name: str = "bge-large-en-v1.5",
    model_dir: str = "BAAI/bge-large-en-v1.5"

@dataclass
class LLMConfig:
    provider: str = "hf"  # "api" | "hf" 
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    temperature: float = 0.6
    top_p: float = 0.9
    max_seq_len: int = 1024
    max_gen_len: int = 256

@dataclass
class PromptConfig:
    suffix: List[str] = field(default_factory=lambda: [
        "context: ", 
        "question: ", 
        "answer:"
    ])
    adhesive: str = "\n"

@dataclass
class RetrievalConfig:
    method: str = "similarity_score_threshold"  # "similarity_score_threshold" | "mmr"
    rerank: str = 'BAAI/bge-reranker-large'
    adhesive: str = "\n\n"
    params: Dict[str, Any] = field(default_factory=lambda: {
        "k": 4,
        "score_threshold": 0.75
    })
    

@dataclass
class ExpConfig:
    output_dir: str = "./exp/demo/"


@dataclass
class BaseConfig:
    datastorage: DataStorageConfig = DataStorageConfig(
        raw_data_dir = ["./data/wikitxt"],
        tool = "chroma"  # "chroma" | "faiss"
    )
    chunk: ChunkConfig = ChunkConfig(
        method = 'recursive',
        params = {
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    )
    embedding: EmbeddingConfig = EmbeddingConfig(
        provider="hf",
        model_name = "all-MiniLM-L6-v2",
        model_dir = "all-MiniLM-L6-v2"
    )
    llm: LLMConfig = LLMConfig(
        provider = "hf",  # "api" | "hf" 
        model_name = "meta-llama/Llama-2-7b-chat-hf",
        temperature = 0.6,
        top_p = 0.9,
        max_seq_len = 1024,
        max_gen_len = 256
    )
    prompt: PromptConfig = PromptConfig(
        suffix=["context: ", "question: ", "answer:"],
        adhesive="\n"
    )
    retrieval: RetrievalConfig = RetrievalConfig(
        method = "mmr",
        rerank = None,
        adhesive = "\n\n",
        params = {
            "k": 4,
            "fetch_k": 40
        }
    )
    expconfig: ExpConfig = ExpConfig(
        output_dir = "./exp/demo/"
    )