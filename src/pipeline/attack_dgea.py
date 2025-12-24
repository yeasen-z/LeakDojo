from src.components import OpenAILLM, LLMQueryRewriter, VectorRetriever, RerankerManager, LLMHybridExtractor, SimplePromptConstructor
from src.skuas import DGEAQueryGenerator, Find_Dissimilar_Vector
from src.pipeline import RAGPipeline
import os
import json
from tqdm import tqdm
import time
from .utils import setup, chunked
import sys
import torch
import pandas as pd

RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"

import random
random.seed(42)

def count_lines(filepath):
    """计算 JSONL 文件中的行数。"""
    if not os.path.exists(filepath):
        return 0
    
    count = 0
    # 使用 'rb' 模式并迭代读取，效率更高
    with open(filepath, 'rb') as f:
        # 统计换行符数量
        count = sum(1 for _ in f)
    return count

def AtkDGEAPipeline(cfg, args):

    llm, llm_tool, intent_filter, output_filter, query_rewriter, retriever, reranker, extractor, constructor = setup(cfg, args)

    rag_pipeline = RAGPipeline(llm, 
                               query_rewriter, 
                               retriever, 
                               reranker, 
                               extractor, 
                               constructor, 
                               intent_filter, output_filter,
                               cfg, args)
    
    vectors_num = args.attack_num

    output_dir = cfg.generate_exp_path(args.llm_model)
    os.makedirs(output_dir, exist_ok=True)
    jsonl_filename = cfg.generate_exp_filename(args, "dgea_attack")
    save_path = os.path.join(output_dir, jsonl_filename)

    dgea_query_generator = DGEAQueryGenerator(llm_tool, vectors_num=vectors_num, device=args.device)
    
    embedding_space = []
    # start_index = 0
    start_index = count_lines(save_path)
    print(f"检测到上次运行已保存 {YELLOW}{start_index}{RESET} 条记录。")
    print(f"本次运行将从向量索引 {YELLOW}{start_index}{RESET} 开始继续。")

    pbar = tqdm(total=vectors_num-start_index, desc="DGEA Extraction")

    perturbed_sentence = dgea_query_generator.prefix

    for index in range(start_index, vectors_num):
        print(f'Starting new vector {index}')
        time_start = time.time()

        if index == 0:
            target_embedding = torch.tensor(dgea_query_generator.Vectors[index]).to(args.device)
        else:
            # target_embedding = Find_Dissimilar_Vector(embedding_space, vector_length=len(dgea_query_generator.Vectors[index]))
            # get_next_target
            target_embedding = dgea_query_generator.get_next_target(
                index=index,
                embedding_space=embedding_space,
                last_query=perturbed_sentence,
                embedding_model=dgea_query_generator.embedding_model,
                vector_dim=len(dgea_query_generator.Vectors[0]),
                device=args.device
            )

        if not isinstance(target_embedding, torch.Tensor):
            target_embedding = torch.tensor(
                target_embedding, dtype=torch.float32, device=args.device
            )
        else:
            target_embedding = target_embedding.to(args.device)

        print(
            f"[Target vector {index}]",
            "finite =", torch.isfinite(target_embedding).all().item(),
            "norm =", torch.norm(target_embedding).item(),
            "min =", target_embedding.min().item(),
            "max =", target_embedding.max().item()
        )
        
        perturbed_suffix, best_loss, best_embedding = dgea_query_generator.gcqAttack(target_embedding, iterations=3, topk=128)
        perturbed_sentence = dgea_query_generator.prefix + ' ' + perturbed_suffix
        print("Perturbed sentence:", perturbed_sentence)
        print("Cosine:", 1 - best_loss)

        (cleaned_batch_queries, contexts, doc_ids, rag_prompt, answers, reasons, rewritten_queries_list, extracted_contexts) = \
            rag_pipeline.run([perturbed_sentence])
        
        result_record = {
            "id": str(index), # 迭代次数作为唯一标识符
            "query_with_template": perturbed_sentence,
            "cleaned_query": cleaned_batch_queries,
            "rewritten_queries": rewritten_queries_list[0] if args.rewriter else [None],
            "contexts": contexts[0],
            "doc_ids": doc_ids[0],
            "extract_contexts": extracted_contexts[0] if args.extractor else [],
            "prompt": rag_prompt[0],
            "answer": answers[0],
            "reason": reasons[0] if args.reasoning else None
        }
        pbar.update(1)
        with open(save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_record, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())

        contents = dgea_query_generator.extract_or_fetch_content(answers[0])
        embedding_space = dgea_query_generator.embed_and_store_unique_contents(contents, embedding_space)

        time_end = time.time()
        time_taken = time_end - time_start
        print(f'Time taken for this vector is {time_taken}')

    print(f"\n[SUCCESS] All results saved to {save_path}")

    return save_path
