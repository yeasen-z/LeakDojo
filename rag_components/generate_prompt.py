from typing import List
from langchain.schema import BaseRetriever
import json
import os

from configs import VectorBaseConfig
from .build_vector_retriever import vector_retrieved_contexts
from .llm_local_inference import run_llm

def query_rewriter(cfg: VectorBaseConfig, query: str, question: str, n: int, model: str, device: str) -> str:
    """
    输入 query，返回多个改写后的 query
    """
    prompt = f"""
你是一个查询改写助手。请根据用户的问题，生成 {n} 个逻辑连贯、不同且相关的检索查询，
希望他能够覆盖更多的相关内容和表述方式。

用户问题: "{query}"
请直接输出改写后的查询，每个一行。
"""

    response = run_llm(cfg, [prompt])[0]

    rewrites = [
        line.strip("-• ").strip()
        for line in response.split("\n")
        if line.strip()
    ]
    return rewrites[:n]

def get_prompts(cfg: VectorBaseConfig, retriever: BaseRetriever, questions: List[str], device: str) -> List[str]:
    '''
    The prompt is consisted by:
        f'{suffix[0]}{united context}{template_adhesive}{suffix[1]}{question}{template_adhesive}{suffix[2]}'
    '''
    prompts = []

    contexts, doc_ids = vector_retrieved_contexts(cfg, questions, retriever, device=device)
    for context, question in zip(contexts, questions):
        prompt = f"{cfg.prompt.suffix[0]}{cfg.retrieval.adhesive.join(context)}{cfg.prompt.adhesive}{cfg.prompt.suffix[1]}{question}{cfg.prompt.adhesive}{cfg.prompt.suffix[2]}"
        prompts.append(prompt)

    with open(os.path.join(cfg.expconfig.output_dir, 'question.json'), 'w', encoding='utf-8') as f_q:
        json.dump(questions, f_q, ensure_ascii=False, indent=2)
    with open(os.path.join(cfg.expconfig.output_dir, 'prompts.json'), 'w', encoding='utf-8') as f_p:
        json.dump(prompts, f_p, ensure_ascii=False, indent=2)
    with open(os.path.join(cfg.expconfig.output_dir, 'context.json'), 'w', encoding='utf-8') as f_c:
        json.dump(contexts, f_c, ensure_ascii=False, indent=2)
    with open(os.path.join(cfg.expconfig.output_dir, 'doc_ids.json'), 'w', encoding='utf-8') as f:
        json.dump(doc_ids, f, ensure_ascii=False, indent=2)
        
    return prompts

