
from src import InfoDepthEvaluator, calculate_diversity_enhanced_score, extract_scores_from_json
import json
import numpy as np

def jsonl_results_loader(save_path,num_records=500):
    """加载 JSONL 格式的结果文件"""
    results = []
    with open(save_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            results.append(record)
    return results[0:num_records]

re_path = "results/nfcorpus/qwen3-32b/R__bge-large-en-v1_5_k10-RR__bge-reranker-large_n5-EX__bge-large-en-v1_5/WBTQ_RW-1_RR-1_EX-1_IF-0_OF-0_none_0.jsonl"
print(re_path)

data_jsonl = jsonl_results_loader(re_path)

data={
        "queries": [item['cleaned_query'] for item in data_jsonl],
        "answers": [item['answer'] for item in data_jsonl],
        "contexts": [item['contexts'] for item in data_jsonl]
    }

infodepth_evaluator = InfoDepthEvaluator(model="gpt-4.1-mini", checkpoint_path=re_path.replace("_none_0.jsonl", "_eval_infodepth.jsonl"))

a,b = infodepth_evaluator.run(data["queries"], data["answers"])

state_now = infodepth_evaluator.state

print(np.mean(state_now["scores"]))