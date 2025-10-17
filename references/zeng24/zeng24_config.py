from dataclasses import dataclass, field
from typing import Dict, Any, List
import os
import re


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
    reasoning: bool = False
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
        "k": 10,
        "n": 3,
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
        reasoning = False,
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
            "k": 10,
            "n": 3,
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


@dataclass
class Zeng24ChatDoctor(VectorBaseConfig):
    # 只覆盖需要修改的部分
    datastorage: vDataStorageConfig = vDataStorageConfig(
        data_name="chatdoctor",
        raw_data_dir = ["./data/chatdoctor"],
        tool = "vector-chroma"
    )
    chunk: vChunkConfig = vChunkConfig(
        params={}
    )
    embedding: vEmbeddingConfig = vEmbeddingConfig(
        provider="hf",
        model_name = "bge-large-en-v1.5",
        model_dir = "BAAI/bge-large-en-v1.5"
    )
    llm: vLLMConfig = vLLMConfig(
        provider = "hf",  # "api" | "hf" 
        # model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/shakechen/Llama-2-7b-chat-hf",
        # model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/ydyajyA/Llama-2-13b-chat-hf",
        # model_name = "./Models/qwen2.5-14B-instruct-1m",
        # model_name = "./Models/Qwen2.5-0.5B-Instruct",
        # model_name = "./Models/Qwen2.5-1.5B-Instruct",
        # model_name = "./Models/Qwen2.5-3B-Instruct",
        # model_name = "./Models/Qwen2.5-7B-Instruct",
        # model_name = "./Models/Qwen2.5-14B-Instruct",
        # model_name = "./Models/Qwen2.5-32B-Instruct",
        # model_name = "./Models/gemma-3-1b-it",
        # model_name = "./Models/gemma-3-27b-it",
        model_name= "./Models/Qwen3-4B",
        reasoning = True,
        max_seq_len = 1024,
        max_gen_len = 1024,
        temperature = 0,
        top_p = 1,
        vllm_parallel_size = 4,
        vllm_gpu_memory_utilization = 0.9
    )
    prompt: vPromptConfig = vPromptConfig(
        suffix=["context: ", "question: ", "answer:"],
        adhesive="\n"
    )
    retrieval: vRetrievalConfig = vRetrievalConfig(
        method="similarity_score_threshold",
        rerank = 'BAAI/bge-reranker-large',
        adhesive = "\n\n",
        params={
            "k": 3,
            "n": 3,
            "score_threshold": 0.5
        }
    )


@dataclass
class Zeng24fiqa(VectorBaseConfig):
    # 只覆盖需要修改的部分
    datastorage: vDataStorageConfig = vDataStorageConfig(
        data_name="fiqa",
        raw_data_dir = ["./data/fiqa"],
        tool = "vector-chroma"
    )
    chunk: vChunkConfig = vChunkConfig(
        params={}
    )
    embedding: vEmbeddingConfig = vEmbeddingConfig(
        provider="hf",
        # model_name = "all-MiniLM-L6-v2",
        # model_dir = "./Models/all-MiniLM-L6-v2",
        # model_name = "all-mpnet-base-v2",
        # model_dir = "./Models/all-mpnet-base-v2",
        model_name = "bge-large-en-v1.5",
        model_dir = "./Models/BAAI-bge-large-en-v1.5"
    )
    llm: vLLMConfig = vLLMConfig(
        provider = "hf",  # "api" | "hf"
        # model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/shakechen/Llama-2-7b-chat-hf",
        # model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/ydyajyA/Llama-2-13b-chat-hf",
        # model_name = "./Models/qwen2.5-14B-instruct-1m",
        # model_name = "./Models/Qwen2.5-0.5B-Instruct",
        # model_name = "./Models/Qwen2.5-1.5B-Instruct",
        # model_name = "./Models/Qwen2.5-3B-Instruct",
        # model_name = "./Models/Qwen2.5-7B-Instruct",
        # model_name = "./Models/Qwen2.5-14B-Instruct",
        # model_name = "./Models/Qwen2.5-32B-Instruct",
        model_name = "./Models/Qwen3-14B",
        # model_name = "./Models/gemma-3-12b-it",
        # model_name = "./Models/gemma-3-27b-it",
        # model_name= "./Models/Qwen3-32B",
        reasoning = True,
        max_seq_len = 4096,
        max_gen_len = 4096,
        temperature = 0,
        top_p = 1,
        vllm_parallel_size = 2,
        vllm_gpu_memory_utilization = 0.9
    )
    prompt: vPromptConfig = vPromptConfig(
        suffix=["context: ", "question: ", "answer:"],
        adhesive="\n"
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



@dataclass
class Zeng24nq(VectorBaseConfig):
    # 只覆盖需要修改的部分
    datastorage: vDataStorageConfig = vDataStorageConfig(
        data_name="nq",
        raw_data_dir = ["./data/nq"],
        tool = "vector-chroma"
    )
    chunk: vChunkConfig = vChunkConfig(
        params={}
    )
    embedding: vEmbeddingConfig = vEmbeddingConfig(
        provider="hf",
        model_name = "bge-large-en-v1.5",
        model_dir = "BAAI/bge-large-en-v1.5"
    )
    llm: vLLMConfig = vLLMConfig(
        provider = "hf",  # "api" | "hf" 
        # model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/shakechen/Llama-2-7b-chat-hf",
        # model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/ydyajyA/Llama-2-13b-chat-hf",
        # model_name = "./Models/qwen2.5-14B-instruct-1m",
        # model_name = "./Models/Qwen2.5-0.5B-Instruct",
        # model_name = "./Models/Qwen2.5-1.5B-Instruct",
        # model_name = "./Models/Qwen2.5-3B-Instruct",
        # model_name = "./Models/Qwen2.5-7B-Instruct",
        # model_name = "./Models/Qwen2.5-14B-Instruct",
        # model_name = "./Models/Qwen2.5-32B-Instruct",
        model_name = "./Models/gemma-3-1b-it",
        max_seq_len = 1024,
        max_gen_len = 1024,
        temperature = 0,
        top_p = 1,
        vllm_parallel_size = 2,
        vllm_gpu_memory_utilization = 0.9
    )
    prompt: vPromptConfig = vPromptConfig(
        suffix=["context: ", "question: ", "answer:"],
        adhesive="\n"
    )
    retrieval: vRetrievalConfig = vRetrievalConfig(
        method="similarity_score_threshold",
        rerank = 'BAAI/bge-reranker-large',
        adhesive = "\n\n",
        params={
            "k": 3,
            "n": 3,
            "score_threshold": 0.5
        }
    )



@dataclass
class Zeng24scifact(VectorBaseConfig):
    # 只覆盖需要修改的部分
    datastorage: vDataStorageConfig = vDataStorageConfig(
        data_name="scifact",
        raw_data_dir = ["./data/scifact"],
        tool = "vector-chroma"
    )
    chunk: vChunkConfig = vChunkConfig(
        params={}
    )
    embedding: vEmbeddingConfig = vEmbeddingConfig(
        provider="hf",
        # model_name = "bge-large-en-v1.5",
        # model_dir = "BAAI/bge-large-en-v1.5"
        model_name = "all-mpnet-base-v2",
        model_dir = "sentence-transformers/all-mpnet-base-v2"
    )
    llm: vLLMConfig = vLLMConfig(
        provider = "hf",  # "api" | "hf" 
        # model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/shakechen/Llama-2-7b-chat-hf",
        # model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/ydyajyA/Llama-2-13b-chat-hf",
        # model_name = "./Models/qwen2.5-14B-instruct-1m",
        # model_name = "./Models/Qwen2.5-0.5B-Instruct",
        # model_name = "./Models/Qwen2.5-1.5B-Instruct",
        # model_name = "./Models/Qwen2.5-3B-Instruct",
        # model_name = "./Models/Qwen2.5-7B-Instruct",
        model_name = "./Models/Qwen2.5-14B-Instruct",
        # model_name = "./Models/Qwen2.5-32B-Instruct",
        # model_name = "./Models/gemma-3-1b-it",
        max_seq_len = 1024,
        max_gen_len = 1024,
        temperature = 0,
        top_p = 1,
        vllm_parallel_size = 2,
        vllm_gpu_memory_utilization = 0.9
    )
    prompt: vPromptConfig = vPromptConfig(
        suffix=["context: ", "question: ", "answer:"],
        adhesive="\n"
    )
    retrieval: vRetrievalConfig = vRetrievalConfig(
        method="similarity_score_threshold",
        # rerank = 'BAAI/bge-reranker-large',
        rerank = "cross-encoder/ms-marco-MiniLM-L6-v2",
        # rerank = None,
        adhesive = "\n\n",
        params={
            "k": 3,
            "n": 3,
            "score_threshold": 0.1
        },
        # method = "mmr",
        # # rerank = 'BAAI/bge-reranker-large',
        # rerank = "cross-encoder/ms-marco-MiniLM-L6-v2",
        # # rerank = None,
        # adhesive = "\n\n",
        # params = {
        #     "k": 3,
        #     "n": 3,
        #     "fetch_k": 40
        # }
    )


@dataclass
class Zeng24arguana(VectorBaseConfig):
    # 只覆盖需要修改的部分
    datastorage: vDataStorageConfig = vDataStorageConfig(
        data_name="arguana",
        raw_data_dir = ["./data/arguana"],
        tool = "vector-chroma"
    )
    chunk: vChunkConfig = vChunkConfig(
        params={}
    )
    embedding: vEmbeddingConfig = vEmbeddingConfig(
        provider="hf",
        model_name = "bge-large-en-v1.5",
        model_dir = "BAAI/bge-large-en-v1.5"
    )
    llm: vLLMConfig = vLLMConfig(
        provider = "hf",  # "api" | "hf" 
        # model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/shakechen/Llama-2-7b-chat-hf",
        # model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/ydyajyA/Llama-2-13b-chat-hf",
        # model_name = "./Models/qwen2.5-14B-instruct-1m",
        # model_name = "./Models/Qwen2.5-0.5B-Instruct",
        # model_name = "./Models/Qwen2.5-1.5B-Instruct",
        # model_name = "./Models/Qwen2.5-3B-Instruct",
        # model_name = "./Models/Qwen2.5-7B-Instruct",
        # model_name = "./Models/Qwen2.5-14B-Instruct",
        # model_name = "./Models/Qwen2.5-32B-Instruct",
        model_name = "./Models/gemma-3-1b-it",
        max_seq_len = 1024,
        max_gen_len = 1024,
        temperature = 0,
        top_p = 1,
        vllm_parallel_size = 2,
        vllm_gpu_memory_utilization = 0.9
    )
    prompt: vPromptConfig = vPromptConfig(
        suffix=["context: ", "question: ", "answer:"],
        adhesive="\n"
    )
    retrieval: vRetrievalConfig = vRetrievalConfig(
        method="similarity_score_threshold",
        rerank = 'BAAI/bge-reranker-large',
        adhesive = "\n\n",
        params={
            "k": 3,
            "n": 3,
            "score_threshold": 0.5
        }
    )