import fire
import warnings
import json
import os
import torch

from configs import VectorBaseConfig
from .rag_utils import get_llm_output_file, get_llm_model_hf_vllm
from vllm import LLM, SamplingParams


def summarization():
    pass


def run_llm(cfg: VectorBaseConfig, all_prompts):
    llm = get_llm_model_hf_vllm(cfg)

    answers = []
    params = SamplingParams(
        max_tokens=cfg.llm.max_gen_len,      # 直接设生成长度
    )
    for prompt in all_prompts:
        outputs = llm.generate([prompt], params)
        generated_text = outputs[0].outputs[0].text.strip()
        answers.append(generated_text)

    with open(os.path.join(cfg.expconfig.output_dir, get_llm_output_file(cfg)), 'w', encoding='utf-8') as f_a:
        json.dump(answers, f_a, ensure_ascii=False, indent=2)
    
    return answers