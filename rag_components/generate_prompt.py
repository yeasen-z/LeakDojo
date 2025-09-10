from typing import List
from langchain.schema import BaseRetriever
import json
import os


from configs import VectorBaseConfig
from .build_vector_retriever import vector_retrieved_contexts


def get_prompts(cfg: VectorBaseConfig, retriever: BaseRetriever, questions: List[str], device: str) -> List[str]:
    '''
    The prompt is consisted by:
        f'{suffix[0]}{united context}{template_adhesive}{suffix[1]}{question}{template_adhesive}{suffix[2]}'
    '''
    prompts = []

    contexts, sources = vector_retrieved_contexts(cfg, questions, retriever, device=device)
    for context, question in zip(contexts, questions):
        prompt = f"{cfg.prompt.suffix[0]}{cfg.retrieval.adhesive.join(context)}{cfg.prompt.adhesive}{cfg.prompt.suffix[1]}{question}{cfg.prompt.adhesive}{cfg.prompt.suffix[2]}"
        prompts.append(prompt)

    with open(os.path.join(cfg.expconfig.output_dir, 'question.json'), 'w', encoding='utf-8') as f_q:
        json.dump(questions, f_q, ensure_ascii=False, indent=2)
    with open(os.path.join(cfg.expconfig.output_dir, 'prompts.json'), 'w', encoding='utf-8') as f_p:
        json.dump(prompts, f_p, ensure_ascii=False, indent=2)
    with open(os.path.join(cfg.expconfig.output_dir, 'context.json'), 'w', encoding='utf-8') as f_c:
        json.dump(contexts, f_c, ensure_ascii=False, indent=2)
    with open(os.path.join(cfg.expconfig.output_dir, 'sources.json'), 'w', encoding='utf-8') as f:
        json.dump(sources, f, ensure_ascii=False, indent=2)
        
    return prompts