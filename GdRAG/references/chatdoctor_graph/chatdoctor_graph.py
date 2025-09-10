from configs.base_graph import *

@dataclass
class gZeng24ChatDoctor(GraphBaseConfig):
    # 只覆盖需要修改的部分
    datastorage: gDataStorageConfig = gDataStorageConfig(
        data_name="chatdoctor",
        data_region = "medical",
        raw_data_dir = ["./data/chatdoctor"],
        ere_extract_llm = "google/flan-t5-base"  # 可换成 flan-t5-large
    )
    chunk: gChunkConfig = gChunkConfig(
        method='by_two_line_breaks',
        params={}
    )
    embedding: gEmbeddingConfig = gEmbeddingConfig(
        provider="hf",
        model_name = "bge-large-en-v1.5",
        model_dir = "BAAI/bge-large-en-v1.5"
    )
    llm: gLLMConfig = gLLMConfig(
        provider = "hf",  # "api" | "hf" 
        # model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/shakechen/Llama-2-7b-chat-hf",
        model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/ydyajyA/Llama-2-13b-chat-hf",
        temperature = 0.6,
        top_p = 0.9,
        max_seq_len = 1024,
        max_gen_len = 256,
        vllm_parallel_size = 2,
        vllm_gpu_memory_utilization = 0.9
    )
    prompt: gPromptConfig = gPromptConfig(
        suffix=["context: ", "question: ", "answer:"],
        adhesive="\n"
    )
    retrieval: gRetrievalConfig = gRetrievalConfig(
        method="similarity_score_threshold",
        rerank = 'BAAI/bge-reranker-large',
        adhesive = "\n\n",
        params={
            "k": 4,
            "score_threshold": 0.0
        }
    )