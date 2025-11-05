from ragas import SingleTurnSample
import ragas
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy, Faithfulness 
from ragas.llms import llm_factory
# from ragas.embeddings import OpenAIEmbeddings, GoogleEmbeddings, HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from ragas import evaluate
import asyncio
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(base_url="http://localhost:22999/v1",
                api_key="EMPTY", # Can be any string
                model="./Models/Qwen2.5-14B-Instruct", timeout=300.0
                )

evaluator_llm = LangchainLLMWrapper(llm)

embedding_model = HuggingFaceEmbeddings(
    model_name="./Models/BAAI-bge-large-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}

)

from ragas import EvaluationDataset


def load_json_dataset(json_path: str):
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
    return ragas_dataset

# dataset = load_json_dataset("exp/fiqa-chroma/bge-large-en-v1_5-Qwen3-32B/mmr-15-bge-reranker-large-10/BAAI-bge-large-en-v1_5/559ca4/rewr-False_rerank-True_sum-False_wbtq.json")
dataset = load_json_dataset("exp/fiqa-chroma/bge-large-en-v1_5-Qwen3-32B/mmr-15-bge-reranker-large-10/BAAI-bge-large-en-v1_5/cf37be/rewr-True_rerank-True_sum-False_wbtq.json")

# dataset = []

# for query, retrieved_context, response in zip(sample_queries, retrieved_contexts, responses):
#     dataset.append(
#         {
#             "user_input": query,
#             "retrieved_contexts": [rdoc for rdoc in retrieved_context],
#             "response": response,
#         }
#     )

evaluation_dataset = EvaluationDataset.from_list(dataset)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextPrecisionWithoutReference(), ResponseRelevancy(), Faithfulness()],
    llm=evaluator_llm,
    embeddings=embedding_model
)

# result = asyncio.run(context_precision.single_turn_ascore(sample))

print(result)