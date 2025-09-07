from configs.base import *

@dataclass
class Zeng24ChatDoctor(BaseConfig):
    # 只覆盖需要修改的部分
    datastorage: DataStorageConfig = DataStorageConfig(
        raw_data_dir = ["./data/chatdoctor"],
        tool = "chroma"  # "chroma" | "faiss"
    )
    chunk: ChunkConfig = ChunkConfig(
        method='by_two_line_breaks',
        params={}
    )
    embedding: EmbeddingConfig = EmbeddingConfig(
        provider="hf",
        model_name = "bge-large-en-v1.5",
        model_dir = "BAAI/bge-large-en-v1.5"
    )
    llm: LLMConfig = LLMConfig(
        provider = "hf",  # "api" | "hf" 
        # model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/shakechen/Llama-2-7b-chat-hf",
        model_name = "/mnt/data1/workplace/zms/Models/modelscope_cache/models/ydyajyA/Llama-2-13b-chat-hf",
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
        method="similarity_score_threshold",
        rerank = 'BAAI/bge-reranker-large',
        adhesive = "\n\n",
        params={
            "k": 4,
            "score_threshold": 0.0
        }
    )
    expconfig: ExpConfig = ExpConfig(
        output_dir = "./exp/zeng24/"
    )


@dataclass
class Zeng24Wikitxt(BaseConfig):
    datastorage: DataStorageConfig = DataStorageConfig(
        raw_data_dir = ["./data/wikitxt"],
        tool = "chroma"  # "chroma" | "faiss"
    )
    chunk: ChunkConfig = ChunkConfig(
        method='recursive',
        params={
            "chunk_size": 1500,
            "chunk_overlap": 100
        }
    )

