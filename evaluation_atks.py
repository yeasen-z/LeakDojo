from src import evaluate_atk_results


if __name__ == "__main__":
    
    save_path = "./results/fiqa/o4-mini/R__bge-large-en-v1_5_k10-RR__bge-reranker-large_n5-EX__bge-large-en-v1_5/BBQG_RW-1_RR-1_EX-0_IF-0_OF-0_repeat_command_0.jsonl"

    evaluate_atk_results(save_path, num_records=200)