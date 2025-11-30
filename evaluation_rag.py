from src import evaluate_infodepth
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

    
if __name__ == "__main__":
    
    save_path = "./results/fiqa/qwen2_5-14b-instruct/R__bge-large-en-v1_5_k10-RR__bge-reranker-large_n5-EX__bge-large-en-v1_5/IKEA_RW-1_RR-1_EX-0_IF-0_OF-0_repeat_command_0.jsonl"

    data_jsonl = jsonl_results_loader(save_path, 200)
    data={
        "queries": [item['cleaned_query'] for item in data_jsonl],
        "answers": [item['answer'] for item in data_jsonl],
        "contexts": [item['contexts'] for item in data_jsonl]
    }

    judge_out, scores = evaluate_infodepth(data["queries"], data["answers"])
    # print(judge_out)
    print(np.mean(scores))
    with open(save_path.replace("_none_0.jsonl", "_infodepth.json"), "w", encoding="utf-8") as f:
        json.dump({
            "judge_out": judge_out
        }, f, ensure_ascii=False, indent=2)