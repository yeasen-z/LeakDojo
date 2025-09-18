from dataclasses import dataclass, field
from typing import Dict, Any, List
import os

@dataclass
class vDataStorageConfig:
    data_name: str = "chatdoctor"
    raw_data_dir: List[str] = field(default_factory=List)
    tool: str = "vector-chroma"  # # "vector-chroma"

@dataclass
class vChunkConfig:
    params: Dict[str, Any] = field(default_factory=lambda: {
        "chunk_size": 1000,
        "chunk_overlap": 200
    })

@dataclass
class vEmbeddingConfig:
    provider: str = "hf"  # "hf" | "openai"
    model_name: str = "bge-large-en-v1.5",
    model_dir: str = "BAAI/bge-large-en-v1.5"

@dataclass
class vLLMConfig:
    provider: str = "hf"  # "api" | "hf" 
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    vllm_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    temperature: float = 0.6
    top_p: float = 0.9
    max_seq_len: int = 1024
    max_gen_len: int = 256

@dataclass
class vPromptConfig:
    suffix: List[str] = field(default_factory=lambda: [
        "context: ", 
        "question: ", 
        "answer:"
    ])
    adhesive: str = "\n"

@dataclass
class vRetrievalConfig:
    method: str = "similarity_score_threshold"  # "similarity_score_threshold" | "mmr"
    rerank: str = 'BAAI/bge-reranker-large'
    adhesive: str = "\n\n"
    params: Dict[str, Any] = field(default_factory=lambda: {
        "k": 2,
        "score_threshold": 0.75
    })


@dataclass
class vExpConfig:
    output_dir: str = "./exp/demo/"


@dataclass
class VectorBaseConfig:
    datastorage: vDataStorageConfig = vDataStorageConfig(
        data_name = "chatdoctor",
        raw_data_dir = ["./data/chatdoctor"],
        tool = "vector-chroma"
    )
    chunk: vChunkConfig = vChunkConfig(
        params = {
            "chunk_size": 1000,
            "chunk_overlap": 200
        }
    )
    embedding: vEmbeddingConfig = vEmbeddingConfig(
        provider="hf",
        model_name = "all-MiniLM-L6-v2",
        model_dir = "all-MiniLM-L6-v2"
    )
    llm: vLLMConfig = vLLMConfig(
        provider = "hf",  # "api" | "hf" 
        model_name = "meta-llama/Llama-2-7b-chat-hf",
        max_seq_len = 1024,
        max_gen_len = 256,
        vllm_parallel_size = 2,
        vllm_gpu_memory_utilization = 0.9
    )
    prompt: vPromptConfig = vPromptConfig(
        suffix=["context: ", "question: ", "answer:"],
        adhesive="\n"
    )
    retrieval: vRetrievalConfig = vRetrievalConfig(
        method = "mmr",
        rerank = None,
        adhesive = "\n\n",
        params = {
            "k": 2,
            "fetch_k": 40
        }
    )

    @property
    def expconfig(self) -> vExpConfig:
        """动态生成实验输出目录"""
        dataset = os.path.basename(self.datastorage.data_name)  # chatdoctor
        store = self.datastorage.tool                           # vector-chroma
        embed = self.embedding.model_name.replace(".","_")                       # bge-large-en-v1.5  -> bge-large-en-v1_5
        llm = os.path.basename(self.llm.model_name).replace(".","_")             # Llama-2-13b-chat-hf
        k = self.retrieval.params.get("k", 2)                                    # 2
        retrieved_method = self.retrieval.method
        reranker = self.retrieval.rerank if self.retrieval.rerank else "no-rerank"

        save_dir = f"./exp/{dataset}/{store}/{embed}-{llm}/{retrieved_method}-{k}-{reranker}/"
        return vExpConfig(output_dir=save_dir)
