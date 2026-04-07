from ragas import SingleTurnSample
import ragas
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy, Faithfulness 
from ragas.llms import llm_factory
# from ragas.embeddings import OpenAIEmbeddings, GoogleEmbeddings, HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import openai
from ragas import evaluate
import asyncio
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_openai import ChatOpenAI
import json

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# conda activate ragas_eva
llm = ChatOpenAI(
        base_url="https://aihubmix.com/v1",
        api_key="YOUR_API_KEY_HERE",
        model="gpt-4.1-mini", timeout=300.0
)

evaluator_llm = LangchainLLMWrapper(llm)

embedding_model = HuggingFaceEmbeddings(
    model_name="./Models/BAAI/bge-large-en-v1.5",
    model_kwargs = {'device': 'cuda:1'},
    encode_kwargs={"normalize_embeddings": True}
)

from ragas import EvaluationDataset

def jsonl_results_loader(save_path):
    """加载 JSONL 格式的结果文件"""
    results = []
    with open(save_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            results.append(record)

    # 转换为 RAGAS 格式
    ragas_dataset = []
    for i in range(len(results)):
        item = {
            "user_input": results[i]["cleaned_query"],
            # "retrieved_contexts": results[i]["contexts"] if "contexts" in results[i] else [],
            "retrieved_contexts": results[i]["extract_contexts"] if results[i]["extract_contexts"] !=[] else results[i]["contexts"],
            "response": results[i]["answer"] if "answer" in results[i] else ""
        }
        ragas_dataset.append(item)
    return ragas_dataset


path_dir = "results/nfcorpus/DeepSeek-V3/R__bge-large-en-v1_5_k10-RR__bge-reranker-large_n5-EX__bge-large-en-v1_5/WBTQ_RW-1_RR-1_EX-1_IF-0_OF-0_none_0.jsonl"

print(path_dir)


dataset = jsonl_results_loader(path_dir)

# print(dataset[0])

evaluation_dataset = EvaluationDataset.from_list(dataset)

result = evaluate(
    dataset=evaluation_dataset,
    # metrics=[LLMContextPrecisionWithoutReference(), ResponseRelevancy(), Faithfulness()],
    metrics=[ResponseRelevancy(), Faithfulness()],
    llm=evaluator_llm,
    embeddings=embedding_model,
    allow_nest_asyncio=False
)

print(path_dir)

print(result)

with open(path_dir.replace("_none_0.jsonl", "_eval_ragas_gpt41mini.json"), "w", encoding="utf-8") as f:
    json.dump(result.scores, f, ensure_ascii=False, indent=2)