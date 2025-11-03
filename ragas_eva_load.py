import asyncio
import os
import json
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy, Faithfulness 
from openai import OpenAI
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

def load_json_dataset(json_path: str) -> Dataset:
    # 读取原始 json 文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # 转换为 RAGAS 格式
    ragas_dataset = []
    for i in range(len(data["queries"])):
        item = {
            "user_input": data["queries"][i],
            "retrieved_contexts": data["contexts"][i] if "contexts" in data else [],
            "response": data["answers"][i] if "answers" in data else ""
        }
        ragas_dataset.append(item)
    return Dataset.from_list(ragas_dataset)

dataset = load_json_dataset("exp/fiqa-chroma/bge-large-en-v1_5-Qwen2_5-7B-Instruct/mmr-15-bge-reranker-large-10/BAAI-bge-large-en-v1_5/e3c1d1/rewr-False_rerank-True_sum-False_wbtq.json")

# vllm serve ./Models/Qwen2.5-14B-Instruct --port 22999 --max-num-seqs 3
llm = ChatOpenAI(base_url="http://localhost:22999/v1",
                api_key="lm-studio", # Can be any string
                model="./Models/Qwen2.5-14B-Instruct"
                )

evaluator_llm = LangchainLLMWrapper(llm)

embedding_model = HuggingFaceEmbeddings(
    model_name="./Models/BAAI-bge-large-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}

)

from ragas.run_config import RunConfig
config = RunConfig(
    timeout=120,           # 超时时间（秒）
    max_retries=2,         # 自动重试次数
)


metric_todo = [
    LLMContextPrecisionWithoutReference(),
    ResponseRelevancy(),
    Faithfulness()
]

results_dodone = []

for m in metric_todo:
    print(f"Evaluating with metric: {m.__class__.__name__}")
    result = evaluate(
        dataset=dataset,
        metrics=[m],
        llm=evaluator_llm,
        embeddings=embedding_model,
        run_config=config,             # ✅ 超时与并发控制
        batch_size=8,                 # 小批量处理，节省显存
    )
    print(result)
    results_dodone.append(result)

print("Final Results:")
for r in results_dodone:
    print(r)
