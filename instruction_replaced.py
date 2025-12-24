
from src import AtkStaticPipeline, AtkIKEAPipeline, AtkRTFPipeline, AtkPoRPipeline, AtkDGEAPipeline
from src import VectorRetriever
import argparse
import json
import configs
from tqdm import tqdm

import sys
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"

import random
random.seed(42)


path_save= "results/scifact/DeepSeek-V3/R__bge-large-en-v1_5_k10-RR__bge-reranker-large_n5-EX__bge-large-en-v1_5/TGTB_RW-1_RR-1_EX-0_IF-1_OF-0_tgtb.jsonl"

save_path = "results/scifact/DeepSeek-V3/R__bge-large-en-v1_5_k10-RR__bge-reranker-large_n5-EX__bge-large-en-v1_5/TGTB_RW-1_RR-1_EX-0_IF-1_OF-0_tgtb_reranker_role_play.jsonl"

import json
import re
from pathlib import Path

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


loaded_records = load_jsonl(path_save)


loaded_records[0]


with open("attack_shop/adv_strings/collection.json", "r", encoding="utf-8") as f:
    template_shop = json.load(f)

template = template_shop["reranker_role_play"]["en_strings"][0]
ad_suf_name = "reranker_role_play"

def chunked(iterable, batch_size):
    """把列表按 batch_size 分块"""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

new_queries = []
for i in range(len(loaded_records)):
    print(loaded_records[i]['id'])
    new_query = template.format(text=loaded_records[i]['cleaned_query'])
    print(new_query)
    new_queries.append(new_query)


total_queries_to_process = len(new_queries)
total_batches = (total_queries_to_process + 2 - 1) // 2
    
batch_iterator = tqdm(
        chunked(new_queries, 2),
        total=total_batches,
        desc=f"Processing Batches ({0} finished)"
    )

with open(save_path, "a", encoding="utf-8") as f:
    for batch_idx, batch_items in enumerate(batch_iterator):
        
        batch_ids = [item['id'] for item in batch_items]
        batch_queries_withtemplate = [item['query'] for item in batch_items]
        clean_queries = [s.replace(template, "") for s in batch_queries_withtemplate]
        
        current_global_idx = 0 + batch_idx * 2
        batch_iterator.set_description(
            f"{RED}Processing Batch {batch_idx+1}/{total_batches} (Q_idx: {current_global_idx}){RESET}"
        )

        # --- 3. 调用 RAG 流水线处理当前批次 ---
        (cleaned_batch_queries, contexts, doc_ids, prompt, answers, reasons, rewritten_queries_list, extracted_contexts) = \
            rag_pipeline.run(batch_queries_withtemplate)

        # --- 4. 逐条构建 JSON 对象并追加写入 JSONL 文件 ---
        for i in range(len(batch_ids)):
            result_record = {
                "id": batch_ids[i], # 关键：使用问题 ID 作为唯一标识符
                "adversarial_template": adversarial_template,
                # "query": clean_queries[i],
                "query_with_template": batch_queries_withtemplate[i],
                "cleaned_query": cleaned_batch_queries[i],
                "rewritten_queries": rewritten_queries_list[i] if args.rewriter else [None],
                "contexts": contexts[i],
                "doc_ids": doc_ids[i],
                "extract_contexts": extracted_contexts[i] if args.extractor else [],
                "prompt": prompt[i],
                "answer": answers[i],
                "reason": reasons[i] if args.reasoning else None
            }
            
            # 写入一行 JSONL
            f.write(json.dumps(result_record, ensure_ascii=False) + '\n')
        
        # 强制将数据写入磁盘，防止断电丢失
        f.flush()
        os.fsync(f.fileno())


