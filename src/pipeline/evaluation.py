from src.components.scoring import RougeEvaluator, LiteralEvaluator, EmbeddingEvaluator, CrossEncoderEvaluator
import json

def jsonl_results_loader(save_path,num_records=500):
    """加载 JSONL 格式的结果文件"""
    results = []
    with open(save_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            results.append(record)
    return results[0:num_records]

def evaluate_results(save_path, num_records=500):
    """评估攻击的结果"""
    data_jsonl = jsonl_results_loader(save_path, num_records)
    data={
        "doc_ids": [item['id'] for item in data_jsonl],
        "answers": [item['answer'] for item in data_jsonl],
        "contexts": [item['contexts'] for item in data_jsonl]
    }
    print(f"Evaluating {len(data['doc_ids'])} records from {save_path}...")
    
    roge05, ltre50, embde08 = RougeEvaluator(0.5), LiteralEvaluator(50), EmbeddingEvaluator(0.8, device="cuda:8")

    rouge_scores_05 = roge05.evaluate(data["doc_ids"], data["answers"], data["contexts"])
    print("Rouge-L[F1]@0.5")
    print(f"rouge_hit_count: {rouge_scores_05['rouge_hit_count']}, unique_contexts: {rouge_scores_05['unique_contexts']}")
    # literal_scores_50 = ltre50.evaluate(data["doc_ids"], data["answers"], data["contexts"])
    # print(f"Literal Match@50: {literal_scores_50}")
    lll = ltre50.evaluate_rougeL_atks(data["doc_ids"], data["answers"], data["contexts"],rouge_scores_05["atks_ids"])
    print(f"evaluate_rougeL_atks: {lll}")
    embedding_scores_08 = embde08.evaluate(data["doc_ids"], data["answers"], data["contexts"])
    print(f"Embedding Similarity@0.8: {embedding_scores_08}")
    # cee08 = CrossEncoderEvaluator(device="cuda:0")
    # cross_encoder_scores_08 = cee08.evaluate_swf(data["doc_ids"], data["answers"], data["contexts"])
    # print(f"Cross Encoder Similarity@0.8: {cross_encoder_scores_08}")

    eval_save_path = save_path.replace(".jsonl", "_eval.json")
    with open(eval_save_path, "w", encoding="utf-8") as f:
        json.dump({
            "Rouge-L@0.5": rouge_scores_05,
            # "Literal Match@50": literal_scores_50,
            "evaluate_rougeL_atks": lll,
            # "Cross Encoder Similarity@0.8": cross_encoder_scores_08,
            "Embedding Similarity@0.8": embedding_scores_08
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation results to {eval_save_path}")