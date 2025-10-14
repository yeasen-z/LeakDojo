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

MODELS_THINKING_SUPPORT = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]

def run_llm(cfg: VectorBaseConfig, all_prompts):
    client = OpenAI(base_url="http://localhost:22999/v1", api_key="EMPTY")

    def call_api(prompt):
        response = client.chat.completions.create(
            model=cfg.llm.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            max_tokens=cfg.llm.max_gen_len,
        )
        return response.choices[0].message.content.strip()

    def call_api_with_reasoning(prompt):
        response = client.chat.completions.create(
            model=cfg.llm.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            max_tokens=cfg.llm.max_gen_len
        )
        # 获取 content
        content = getattr(response.choices[0].message, "content", "")
        content = content.strip() if content else ""

        # 获取 reasoning_content，如果不存在则返回 ""
        reasoning = getattr(response.choices[0].message, "reasoning_content", "")
        reasoning = reasoning.strip() if reasoning else ""

        return content, reasoning
    
    answers = []
    reasons = []

    if cfg.llm.reasoning and os.path.basename(cfg.llm.model_name) in MODELS_THINKING_SUPPORT:
        print("Using reasoning mode for LLM calls.")
        with ThreadPoolExecutor(max_workers=50) as executor:
            # future_to_prompt = {executor.submit(call_api_with_reasoning, prompt): prompt for prompt in all_prompts}
            results = list(executor.map(call_api_with_reasoning, all_prompts))  # 顺序和输入一样
            for answer, reason in results:
                answers.append(answer)
                reasons.append(reason)
                
        with open(os.path.join(cfg.expconfig.output_dir, get_llm_output_file(cfg)), 'w', encoding='utf-8') as f_a:
            json.dump(answers, f_a, ensure_ascii=False, indent=2)
        with open(os.path.join(cfg.expconfig.output_dir, get_llm_output_file(cfg)).replace('.json', '_reasoning.json'), 'w', encoding='utf-8') as f_r:
            json.dump(reasons, f_r, ensure_ascii=False, indent=2)

    else:
        with ThreadPoolExecutor(max_workers=50) as executor:
            answers = list(executor.map(call_api, all_prompts))  # 顺序和输入一样

        with open(os.path.join(cfg.expconfig.output_dir, get_llm_output_file(cfg)), 'w', encoding='utf-8') as f_a:
            json.dump(answers, f_a, ensure_ascii=False, indent=2)

    return answers