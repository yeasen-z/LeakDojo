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
        model_name = "./Models/Qwen2.5-1.5B-Instruct",
        # model_name = "./Models/Qwen2.5-3B-Instruct",
        # model_name = "./Models/Qwen2.5-7B-Instruct",
        # model_name = "./Models/Qwen2.5-14B-Instruct",
        # model_name = "./Models/Qwen2.5-32B-Instruct",
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
        model_name = "./Models/Qwen2.5-3B-Instruct",
        # model_name = "./Models/Qwen2.5-7B-Instruct",
        # model_name = "./Models/Qwen2.5-14B-Instruct",
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
            "score_threshold": 0.5
        }
    )



@dataclass
class Zeng24msmarco(VectorBaseConfig):
    # 只覆盖需要修改的部分
    datastorage: vDataStorageConfig = vDataStorageConfig(
        data_name="msmarco",
        raw_data_dir = ["./data/msmarco"],
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
        model_name = "./Models/Qwen2.5-1.5B-Instruct",
        # model_name = "./Models/Qwen2.5-3B-Instruct",
        # model_name = "./Models/Qwen2.5-7B-Instruct",
        # model_name = "./Models/Qwen2.5-14B-Instruct",
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
            "score_threshold": 0.5
        }
    )