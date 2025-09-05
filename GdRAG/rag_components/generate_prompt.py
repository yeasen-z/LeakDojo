from typing import List
from langchain.schema import BaseRetriever
import json
import os


from configs import BaseConfig
from .data_retrieval import get_retrieved_contexts


def get_prompts(cfg: BaseConfig, retriever: BaseRetriever, questions: List[str], device: str) -> List[str]:
    '''
    The prompt is consisted by:
        f'{suffix[0]}{united context}{template_adhesive}{suffix[1]}{question}{template_adhesive}{suffix[2]}'
    '''
    prompts = []

    contexts = get_retrieved_contexts(cfg, questions, retriever, device=device)
    for context, question in zip(contexts, questions):
        prompt = f"{cfg.prompt.suffix[0]}{context}{cfg.prompt.adhesive}{cfg.prompt.suffix[1]}{question}{cfg.prompt.adhesive}{cfg.prompt.suffix[2]}"
        prompts.append(prompt)
    
    with open(cfg.expconfig.output_dir + '/question.json', 'w', encoding='utf-8') as f_q:
        json.dump(questions, f_q, ensure_ascii=False, indent=2)
    with open(cfg.expconfig.output_dir + '/prompts.json', 'w', encoding='utf-8') as f_p:
        json.dump(prompts, f_p, ensure_ascii=False, indent=2)
    with open(cfg.expconfig.output_dir + '/context.json', 'w', encoding='utf-8') as f_c:
        json.dump(contexts, f_c, ensure_ascii=False, indent=2)
        
    return prompts