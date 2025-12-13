from src import evaluate_atk_results


if __name__ == "__main__":
    
    # save_path = "results/scifact/qwen2_5-14b-instruct/R__bge-large-en-v1_5_k10-RR__bge-reranker-large_n5-EX__bge-large-en-v1_5/POR_RW-0_RR-0_EX-0_IF-0_OF-0_por_attack.jsonl"

    save_path = "results/enronmail/qwen2_5-14b-instruct/R__bge-large-en-v1_5_k10-RR__bge-reranker-large_n5-EX__bge-large-en-v1_5/POR_RW-0_RR-0_EX-0_IF-0_OF-0_por_attack_not_finished.jsonl"
    evaluate_atk_results(save_path, num_records=200)