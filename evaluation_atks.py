from src import evaluate_atk_results, evaluate_atk_results_rougeL
from pathlib import Path

def parse_setting_from_path(save_path: str):
    p = Path(save_path)

    dataset = p.parts[1]
    generator = p.parts[2]

    filename = p.stem 

    parts = filename.split("_")

    attack = parts[0].lower()

    flag_map = {
        "RW": False,
        "RR": False,
        "EX": False,
    }

    for part in parts:
        for key in flag_map:
            if part.startswith(f"{key}-"):
                flag_map[key] = part.endswith("1")

    setting_now = [
        attack,              
        None,                
        generator,           
        flag_map["RW"],      
        flag_map["RR"],      
        flag_map["EX"],      
        dataset              
    ]

    return setting_now

if __name__ == "__main__":
    
    save_path = "results_nosota/enronmail/DeepSeek-V3/R__bge-large-en-v1_5_k10-RR__bge-reranker-large_n5-EX__bge-large-en-v1_5/POR_RW-0_RR-1_EX-0_IF-0_OF-0_por_attack.jsonl"
    setting_now = parse_setting_from_path(save_path)
    
    # unique_chunks, ly_05, as_num05, rq = evaluate_atk_results(save_path, num_records=50)
    ly_03, as_num03,ly_07, as_num07,ly_09, as_num09=evaluate_atk_results_rougeL(save_path, num_records=200)
    print("Final Results:")
    print("setting_now =", setting_now)
    # print("list =",[unique_chunks, ly_05, as_num05, rq, avg_ss,  ly_03, as_num03, ly_07, as_num07, ly_09, as_num09])