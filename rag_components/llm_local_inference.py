import fire
import warnings
import json
import os
import torch
from tqdm import tqdm

from configs import VectorBaseConfig
from .rag_utils import get_llm_output_file
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


def summarization():
    pass


def run_llm(cfg: VectorBaseConfig, all_prompts):
    client = OpenAI(base_url="http://localhost:8899/v1", api_key="EMPTY")


    def call_api(prompt):
        response = client.chat.completions.create(
            model=cfg.llm.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            max_tokens=cfg.llm.max_gen_len,
        )
        return response.choices[0].message.content.strip()

    answers = []


    with ThreadPoolExecutor(max_workers=50) as executor:
        answers = list(executor.map(call_api, all_prompts))  # 顺序和输入一样

    # for prompt in tqdm(all_prompts, desc="Generating answers", unit="prompt"):
    #     response = client.chat.completions.create(
    #         model=cfg.llm.model_name,
    #         messages=[{"role": "user", "content": prompt}],
    #         temperature=cfg.llm.temperature,
    #         top_p=cfg.llm.top_p,
    #         max_tokens=cfg.llm.max_gen_len,
    #     )
    #     generated_text = response.choices[0].message.content.strip()
    #     answers.append(generated_text)

    with open(os.path.join(cfg.expconfig.output_dir, get_llm_output_file(cfg)), 'w', encoding='utf-8') as f_a:
        json.dump(answers, f_a, ensure_ascii=False, indent=2)
    
    return answers