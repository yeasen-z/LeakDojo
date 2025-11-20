from src import evaluate_results


if __name__ == "__main__":
    # save_path = "results/fiqa/qwen3-32b/R__bge-large-en-v1_5_k15-RR__bge-reranker-large_n10-S__bge-large-en-v1_5/WBTQ_RW-0_RR-0_S-0_raw_en_strings_0.jsonl"
    
    # save_path = "/mnt/data1/workspace/zms/LeakDojo/results/fiqa/qwen3-32b/R__bge-large-en-v1_5_k15-RR__bge-reranker-large_n10-S__bge-large-en-v1_5/WBTQ_RW-0_RR-1_S-0_raw_en_strings_0.jsonl"
    
    # save_path = "/mnt/data1/workspace/zms/LeakDojo/results/fiqa/qwen3-32b/R__bge-large-en-v1_5_k10-RR__bge-reranker-large_n5-S__bge-large-en-v1_5/WBTQ_RW-1_RR-1_S-0_raw_en_strings_0.jsonl"
    
    save_path = "/mnt/data1/workspace/zms/LeakDojo/results/fiqa/qwen3-32b/R__bge-large-en-v1_5_k10-RR__bge-reranker-large_n5-S__bge-large-en-v1_5/WBTQ_RW-0_RR-0_S-0_IF-1_OF-0_code_en_strings_2.jsonl"
    evaluate_results(save_path)