from src.interfaces import QueryGenerator
from typing import List, Union
import json
import random


class WhiteBoxQueryLoader(QueryGenerator):
    """白盒静态的问题加载器，从本地文件加载问题"""

    def __init__(self, filepath: Union[str, List[str]], min_len: int = 20, max_len: int = 250, attack_num: int = 500):
        self.filepath = filepath
        self.attack_num = attack_num
        self.min_len = min_len
        self.max_len = max_len
    
    def _load_and_filter(self, filepath: str) -> List[str]:
        questions = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                questions.append(doc["text"])
        return questions

    def generate(self) -> List[str]:
        if isinstance(self.filepath, list):
            questions = []
            for fp in self.filepath:
                questions.extend(self._load_and_filter(fp))
        else:
            questions = self._load_and_filter(self.filepath)
        
        # 过滤掉过短或过长的问题
        filtered = [
            q.strip() for q in questions
            if self.min_len <= len(q.strip()) <= self.max_len
        ]

        if len(filtered) > self.attack_num:
            filtered = random.sample(filtered, self.attack_num)

        return filtered