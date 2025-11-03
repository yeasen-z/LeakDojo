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


# 创建 OpenAI client 指向本地 vLLM server
# client = OpenAI(base_url="http://localhost:22999/v1", api_key="EMPTY")

# 在 LangChain 中创建 ChatOpenAI 对象
# llm = ChatOpenAI(name="./Models/Qwen2.5-14B-Instruct", client=client)
llm = ChatOpenAI(base_url="http://localhost:22999/v1",
                api_key="lm-studio", # Can be any string
                model="./Models/Qwen2.5-32B-Instruct", timeout=300.0
                )

evaluator_llm = LangchainLLMWrapper(llm)

embedding_model = HuggingFaceEmbeddings(
    model_name="./Models/BAAI-bge-large-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}

)

# sample_queries = [
#     "Which CEO is widely recognized for democratizing AI education through platforms like Coursera?",
#     "Who is Sam Altman?",
#     "Who is Demis Hassabis and how did he gained prominence?",
#     "Who is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's product ecosystem?",
#     "How did Arvind Krishna transformed IBM?",
# ]

# retrieved_contexts = [
#     ["Andrew Ng is the CEO of Landing AI and is widely recognized for democratizing AI education through platforms like Coursera."],
#     ["Sam Altman is the CEO of OpenAI and has played a key role in advancing AI research and development. He strongly advocates for creating safe and beneficial AI technologies."],
#     ["Demis Hassabis is the CEO of DeepMind and is celebrated for his innovative approach to artificial intelligence. He gained prominence for developing systems like AlphaGo that can master complex games."],
#     ["Sundar Pichai is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's vast product ecosystem. His leadership has significantly enhanced user experiences globally."],
#     ["Arvind Krishna is the CEO of IBM and has transformed the company towards cloud computing and AI solutions. He focuses on delivering cutting-edge technologies to address modern business challenges."],
# ]

# responses = [
#     "Andrew Ng is the CEO of Landing AI and is widely recognized for democratizing AI education through platforms like Coursera.",
#     "Sam Altman is the CEO of OpenAI and has played a key role in advancing AI research and development. He strongly advocates for creating safe and beneficial AI technologies.",
#     "Demis Hassabis is the CEO of DeepMind and is celebrated for his innovative approach to artificial intelligence. He gained prominence for developing systems like AlphaGo that can master complex games.",
#     "Sundar Pichai is the CEO of Google and Alphabet Inc., praised for leading innovation across Google's vast product ecosystem. His leadership has significantly enhanced user experiences globally.",
#     "Arvind Krishna is the CEO of IBM and has transformed the company towards cloud computing and AI solutions. He focuses on delivering cutting-edge technologies to address modern business challenges.",
# ]

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

# dataset = load_json_dataset("exp/fiqa-chroma/bge-large-en-v1_5-Qwen2_5-7B-Instruct/mmr-15-bge-reranker-large-10/BAAI-bge-large-en-v1_5/e3c1d1/rewr-False_rerank-True_sum-False_wbtq.json")
dataset = load_json_dataset("exp/fiqa-chroma/bge-large-en-v1_5-Qwen2_5-14B-Instruct/mmr-15-bge-reranker-large-10/BAAI-bge-large-en-v1_5/8bfe19/rewr-True_rerank-True_sum-False_wbtq.json")
# dataset = load_json_dataset("exp/fiqa-chroma/bge-large-en-v1_5-Qwen2_5-32B-Instruct/mmr-15-bge-reranker-large-10/BAAI-bge-large-en-v1_5/c58db0/rewr-False_rerank-True_sum-False_wbtq.json")

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