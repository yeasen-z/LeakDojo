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

    def __init__(self, 
                 model: str = "./Models/Qwen2.5-14B-Instruct", 
                 base_url: str = "http://localhost:22999/v1", 
                 api_key: str = "EMPTY", 
                 reasoning: bool = True,
                 temperature: float = 0.8,
                 top_p: float = 0.9,
                 max_gen_len: int = 4096,
                 max_workers: int = 50):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.reasoning = reasoning
        self.max_workers = max_workers
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _call_api(self, prompt: str) -> str:
        """普通模式调用"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_gen_len,
        )
        return response.choices[0].message.content.strip()

    def _call_api_with_reasoning(self, prompt: str):
        """带 reasoning_content 的模型调用"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_gen_len
        )

        content = getattr(response.choices[0].message, "content", "") or ""
        reasoning = getattr(response.choices[0].message, "reasoning_content", "") or ""

        return content.strip(), reasoning.strip()

    def infer(self, prompt: str) -> str:
        """单条推理（接口定义要求）"""
        if self.reasoning and os.path.basename(self.model) in MODELS_THINKING_SUPPORT:
            answers, reasons = self._call_api_with_reasoning(prompt)
            return answers, reasons
        else:
            return self._call_api(prompt), None

    def batch_infer(self, all_prompts: List[str]) -> List[str]:
        """批量推理，带多线程"""
        answers, reasons = [], []

        reasoning_mode = (
            self.reasoning and
            os.path.basename(self.model) in MODELS_THINKING_SUPPORT
        )

        with ThreadPoolExecutor(max_workers=50) as executor:
            # if reasoning_mode:
            #     results = list(executor.map(self._call_api_with_reasoning, all_prompts))
            # else:
            #     results = list(executor.map(self._call_api, all_prompts))
            results = list(executor.map(self.infer, all_prompts))

    
        for answer, reason in results:
            answers.append(answer)
            reasons.append(reason)

        return answers, reasons