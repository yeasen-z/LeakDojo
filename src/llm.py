import os
import json
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from typing import List
from abc import ABC
from configs import VectorBaseConfig

from .interfaces import LLMManager
from .utils import get_llm_output_file


MODELS_THINKING_SUPPORT = ["Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"]


class OpenAILLM(LLMManager):
    """基于 OpenAI 接口（包括兼容接口，如本地 vllm）的大模型推理类"""

    def __init__(self, cfg: VectorBaseConfig, model: str = "./Models/Qwen2.5-14B-Instruct", base_url: str = "http://localhost:22999/v1", api_key: str = "EMPTY", max_workers: int = 50):
        self.cfg = cfg
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_workers = max_workers
        
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _call_api(self, prompt: str) -> str:
        """普通模式调用"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.llm.temperature,
            top_p=self.cfg.llm.top_p,
            max_tokens=self.cfg.llm.max_gen_len,
        )
        return response.choices[0].message.content.strip()

    def _call_api_with_reasoning(self, prompt: str):
        """带 reasoning_content 的模型调用"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.llm.temperature,
            top_p=self.cfg.llm.top_p,
            max_tokens=self.cfg.llm.max_gen_len
        )

        content = getattr(response.choices[0].message, "content", "") or ""
        reasoning = getattr(response.choices[0].message, "reasoning_content", "") or ""

        return content.strip(), reasoning.strip()

    def infer(self, prompt: str) -> str:
        """单条推理（接口定义要求）"""
        if self.cfg.llm.reasoning and os.path.basename(self.model) in MODELS_THINKING_SUPPORT:
            content, _ = self._call_api_with_reasoning(prompt)
            return content
        else:
            return self._call_api(prompt)

    def batch_infer(self, all_prompts: List[str]) -> List[str]:
        """批量推理，带多线程"""
        answers, reasons = [], []

        reasoning_mode = (
            self.cfg.llm.reasoning and
            os.path.basename(self.model) in MODELS_THINKING_SUPPORT
        )

        with ThreadPoolExecutor(max_workers=50) as executor:
            if reasoning_mode:
                results = list(executor.map(self._call_api_with_reasoning, all_prompts))
                for answer, reason in results:
                    answers.append(answer)
                    reasons.append(reason)
            else:
                answers = list(executor.map(self._call_api, all_prompts))

        # 保存结果
        output_dir = self.cfg.expconfig.output_dir
        os.makedirs(output_dir, exist_ok=True)

        answers_path = os.path.join(output_dir, get_llm_output_file(self.cfg))
        with open(answers_path, "w", encoding="utf-8") as f_a:
            json.dump(answers, f_a, ensure_ascii=False, indent=2)

        if reasoning_mode:
            reasons_path = answers_path.replace(".json", "_reasoning.json")
            with open(reasons_path, "w", encoding="utf-8") as f_r:
                json.dump(reasons, f_r, ensure_ascii=False, indent=2)

        return answers
