import fire
import warnings
import json
import os
import torch

from configs import VectorBaseConfig
from .rag_utils import get_llm_output_file, get_llm_model_hf


def summarization():
    pass


def run_llm(cfg: VectorBaseConfig, all_prompts):
    tokenizer, generator = get_llm_model_hf(cfg)

    answers = []
    for prompt in all_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(generator.device)
        output_ids = generator.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + cfg.llm.max_gen_len,
            do_sample=True,
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            pad_token_id=tokenizer.eos_token_id
        )
        # 去掉 prompt 前缀
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = generated_text[len(prompt):].strip()
        answers.append(generated_text)


    with open(os.path.join(cfg.expconfig.output_dir, get_llm_output_file(cfg)), 'w', encoding='utf-8') as f_a:
        json.dump(answers, f_a, ensure_ascii=False, indent=2)
    
    return answers