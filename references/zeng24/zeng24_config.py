from configs.base_vector import *

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
        model_name = "./Models/Qwen2.5-14B-Instruct",
        # model_name = "./Models/Qwen2.5-32B-Instruct",
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
        # params={
        #     "k": 10,
        #     "n": 3,
        #     "fetch_k": 40
        # }

        # method="similarity_score_threshold",

        rerank = 'BAAI/bge-reranker-large',
        adhesive = "\n\n",
        params={
            "k": 15,
            "n": 10,
            "score_threshold": 0.5
        }
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