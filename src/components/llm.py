import os
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from typing import List
from functools import partial

from src.interfaces import LLMManager


# MODELS_THINKING_SUPPORT = [
#     "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B",
#     "qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b",
#     ]


class OpenAILLM(LLMManager):
    """基于 OpenAI 接口（包括兼容接口，如本地 vllm）的大模型推理类"""

    def __init__(self, 
                 model: str = "./Models/Qwen2.5-14B-Instruct", 
                 base_url: str = "http://localhost:22999/v1", 
                 api_key: str = "EMPTY", 
                 reasoning: bool = False,
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

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=3, timeout=240)
  
    def _call_api(self, prompt: str, temperature: float = None, top_p: float = None, sysprompt: str = None) -> str:
        """普通模式调用"""
        if temperature is None or top_p is None:
            temperature = self.temperature
            top_p = self.top_p
        
        if sysprompt is None:
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                # top_p=top_p,
                # max_tokens=self.max_gen_len,
                stream=True, 
            )
        else:
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sysprompt} if sysprompt else None,
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                # top_p=top_p,
                # max_tokens=self.max_gen_len,
                stream=True, 
            )
        collected_content = []
                
        for chunk in response_stream:
            if not chunk.choices: #已经到了结尾或者当前chunk输出为空
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                collected_content.append(delta.content)

        full_content = "".join(collected_content).strip()

        return full_content
    
    def _call_api_with_reasoning(self, prompt: str, temperature: float = None, top_p: float = None, sysprompt: str = None):
        if temperature is None or top_p is None:
            temperature = self.temperature
            top_p = self.top_p
        if sysprompt is None:
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                # top_p=top_p,
                # max_tokens=self.max_gen_len,
                stream=True, 
            )
        else:
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sysprompt} if sysprompt else None,
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                # top_p=top_p,
                # max_tokens=self.max_gen_len,
                stream=True, 
            )
        collected_content = []
        collected_reasoning = []
        
        # 2. 循环接收流式数据块 (Chunk)
        for chunk in response_stream:
            if not chunk.choices: #已经到了结尾或者当前chunk输出为空
                continue
            delta = chunk.choices[0].delta

            if hasattr(delta, 'content') and delta.content:
                collected_content.append(delta.content)

            r_content = getattr(delta, "reasoning_content", None)
            if r_content:
                collected_reasoning.append(r_content)

        full_content = "".join(collected_content).strip()
        full_reasoning = "".join(collected_reasoning).strip()

        return full_content, full_reasoning

    def infer(self, prompt: str, temperature: float = None, top_p: float = None, sysprompt: str = None) -> str:
        """单条推理（接口定义要求）"""
        if temperature is None or top_p is None:
            temperature = self.temperature
            top_p = self.top_p
        # if self.reasoning and os.path.basename(self.model) in MODELS_THINKING_SUPPORT:
        if self.reasoning:
            answers, reasons = self._call_api_with_reasoning(prompt, temperature, top_p, sysprompt)
            return answers, reasons
        else:
            return self._call_api(prompt, sysprompt), None

    def batch_infer(self, all_prompts: List[str], temperature: float = None, top_p: float = None, sysprompt: str = None) -> List[str]:
        """批量推理，带多线程"""
        if temperature is None or top_p is None:
            temperature = self.temperature
            top_p = self.top_p
        answers, reasons = [], []

        # reasoning_mode = (
        #     self.reasoning and
        #     os.path.basename(self.model) in MODELS_THINKING_SUPPORT
        # )

        partial_infer = partial(self.infer, temperature=temperature, top_p=top_p, sysprompt=sysprompt)

        with ThreadPoolExecutor(max_workers=8) as executor:
            # if reasoning_mode:
            #     results = list(executor.map(self._call_api_with_reasoning, all_prompts))
            # else:
            #     results = list(executor.map(self._call_api, all_prompts))
            results = list(executor.map(partial_infer, all_prompts))

    
        for answer, reason in results:
            answers.append(answer)
            reasons.append(reason)

        return answers, reasons