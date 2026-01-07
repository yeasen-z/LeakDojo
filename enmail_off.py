
from src import AtkStaticPipeline, AtkIKEAPipeline, AtkRTFPipeline, AtkPoRPipeline, AtkDGEAPipeline
from src import VectorRetriever
import argparse
import json
import configs
from src.components import LLMIntentFilter, RougeLResponseFilter
from src.components.llm import OpenAILLM
from src.components import OpenAILLM, LLMQueryRewriter, VectorRetriever, RerankerManager, LLMHybridExtractor, SimplePromptConstructor

import sys
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"

import random
random.seed(42)


import re

def strip_enron_global_clean(raw_text):
    """
    全局扫描并剔除所有符合邮件头、转发块、以及散落在正文中的元数据短句
    """
    # 1. 定义需要严格匹配开头的关键字（通常是元数据）
    strict_header_keys = {
        'Date:', 'From:', 'To:', 'Subject:', 'Mime-Version:', 
        'Content-Type:', 'Content-Transfer-Encoding:', 'X-From:', 
        'X-To:', 'X-cc:', 'X-bcc:', 'X-Folder:', 'X-Origin:', 'X-FileName:',
        'cc:', 'bcc:', 'Re:', 'FW:', 'Fwd:', 'Sent:'
    }

    # 2. 预编译更复杂的模式（处理时间戳、转发分割线等短句）
    # 匹配类似 "05/22/99 11:12 AM" 或 "----- Forwarded by" 或 "----------------------"
    patterns_to_remove = [
        re.compile(r'^\d{2}/\d{2}/\d{2,4}\s+\d{2}:\d{2}\s+[AP]M'), # 时间戳行
        re.compile(r'^-+.*Forwarded by.*-+', re.IGNORECASE),       # 转发分割线
        re.compile(r'^Sent by:.*', re.IGNORECASE),                 # 发送人标记
        re.compile(r'^-{3,}.*'),                                   # 长横线
    ]

    lines = raw_text.split('\n')
    clean_lines = []

    for line in lines:
        stripped_line = line.strip()
        
        # 逻辑 A: 跳过空行
        if not stripped_line:
            continue
            
        # 逻辑 B: 检查是否以任意元数据关键字开头
        if any(stripped_line.startswith(key) for key in strict_header_keys):
            # print(f"[Removed Key]: {stripped_line}")
            continue
            
        # 逻辑 C: 检查是否匹配复杂的短句模式
        if any(pattern.match(stripped_line) for pattern in patterns_to_remove):
            # print(f"[Removed Pattern]: {stripped_line}")
            continue
            
        # 逻辑 D: 额外的启发式过滤——如果一行太短且包含冒号（例如 "Mark Taylor:"）
        # 这种行往往是残留的元数据。
        if len(stripped_line) < 40 and ":" in stripped_line:
            # 排除掉正常的句子（通常以标点结束）
            if not stripped_line.endswith(('.', '?', '!', '"')):
                # print(f"[Removed Short Metadata]: {stripped_line}")
                continue

        # 剩下的才是真正的正文
        clean_lines.append(line)

    # 重新组合
    return "\n".join(clean_lines).strip()


path_save = "results/enronmail/o4-mini/R__bge-large-en-v1_5_k10-RR__bge-reranker-large_n5-EX__bge-large-en-v1_5/TGTB_RW-0_RR-0_EX-0_IF-0_OF-0_tgtb_0.jsonl"


output_path = path_save.replace(".jsonl", "_emailoff.jsonl")


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


llm = OpenAILLM(
                    base_url="https://aihubmix.com/v1",
                    api_key="sk-XWaGp10Cjy2pZfttA8E538967f7f4dA7A463F584C17b63Bf",
                    model="o4-mini"
            )
            
constructor = SimplePromptConstructor()


replaced_num = len(load_jsonl(output_path)) if Path(output_path).exists() else 0


print(f"Resuming from {replaced_num} replaced records.")


with open(output_path, "a", encoding="utf-8") as f_out:
    for i in range(replaced_num, len(loaded_records)):
        record = loaded_records[i]

        contexts = record["contexts"]
        doc_ids = record["doc_ids"]
        query_with_template = record["query_with_template"]
        new_contexts=[]
        for i in loaded_records[0]["contexts"]:
            new_context = strip_enron_global_clean(i)
            new_contexts.append(new_context)

        prompt = constructor.batch_construct([query_with_template], [new_contexts])
        print("Constructed prompt:", prompt[0])

        answers, reasons = llm.batch_infer(prompt)
        
        record["query_with_template"] = query_with_template
        record["contexts"] = new_contexts
        record["prompt"] = prompt[0]
        record["answer"] = answers[0]

        f_out.write(json.dumps(record) + "\n")
        # break





