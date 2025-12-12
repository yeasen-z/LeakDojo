import json
import random
import re
from collections import Counter
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
from langchain_huggingface import HuggingFaceEmbeddings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import textwrap
import tiktoken
from src.interfaces import LLMManager, QueryGenerator
from src.components.utils import get_embed_model

class CCEAQueryGenerator(QueryGenerator):
    """
    生成式反馈驱动探索 (Generative Feedback-Driven Exploration, GFDE) 引擎。
    集成了统一存储、双策略查询生成和文段续写功能。
    """
    def __init__(self, initial_topic: str, similarity_threshold: float = 0.8):
        # 统一存储结构 (替换 KnowledgeStorage 类)
        self._storage: Dict = {
            'query_embeddings': [],          # 历史查询的嵌入向量 (用于多样性检查)
            'covered_concepts': set(),       # 已覆盖的知识概念 (用于全局探索引导)
            'latest_context': f"初始设定：正在探索关于'{initial_topic}'的知识。",
            'similarity_threshold': similarity_threshold
        }
        self.current_query: str = initial_topic
        print(f"✅ GFDE 引擎初始化。初始主题: {initial_topic}")

    # --------------------------------------------------------
    # I. 统一存储与多样性检查 (核心记忆机制)
    # --------------------------------------------------------

    def _mock_get_embedding(self, text: str) -> np.ndarray:
        """【MOCK】模拟将文本转换为嵌入向量 (128维随机向量)。"""
        # TODO: 替换为实际的嵌入模型调用，例如 Sentence Transformer 或 API
        return np.random.rand(128)

    def _check_and_add_query(self, query: str) -> bool:
        """计算查询嵌入并检查新颖性，如果新颖则添加到存储中。"""
        new_embed = self._mock_get_embedding(query)
        embeddings = self._storage['query_embeddings']
        threshold = self._storage['similarity_threshold']
        
        # 多样性检查
        if embeddings:
            # 简化：使用余弦相似度（向量点积，如果向量已归一化）
            similarities = [np.dot(new_embed, old_embed) for old_embed in embeddings]
            max_similarity = max(similarities) if similarities else 0
            
            if max_similarity >= threshold:
                return False  # 相似度过高，被拒绝

        # 接受并存储
        self._storage['query_embeddings'].append(new_embed)
        return True

    def _update_covered_knowledge(self, concepts: List[str]):
        """根据答案更新已覆盖的知识点，用于指导全局探索。"""
        self._storage['covered_concepts'].update(concepts)

    # --------------------------------------------------------
    # II. 查询生成与筛选 (双策略集成)
    # --------------------------------------------------------

    def _generate_global_queries(self) -> List[str]:
        """【策略 A: 全局探索】基于知识稀疏性强制切换主题。"""
        # TODO: 替换为实际的 LLM 或图谱稀疏区域查询
        unexplored_topics = [
            "量子计算的加密挑战", 
            "古希腊哲学的伦理核心", 
            "全球供应链的弹性策略"
        ]
        # 过滤掉已覆盖的，确保探索新领域
        candidates = [t for t in unexplored_topics if t not in self._storage['covered_concepts']]
        return candidates[:2] if candidates else ["生态系统的能量流动"]

    def _generate_local_queries(self) -> List[str]:
        """【策略 B: 局部反馈】基于当前文段上下文反向生成问题。"""
        context = self._storage['latest_context']
        if not context:
            return ["什么是知识图谱？"]
        
        # TODO: 替换为基于 LLM 的反向生成，分析 context 并提出未决问题
        # 假设 LLM 分析上下文，并提出能够"丰富、扩展或解决"当前文段中未决问题的查询
        mock_questions = [
            f"基于上文讨论的机制，其在**现实应用中**的**主要局限性**是什么?",
            f"请**对比**与上文中提到的概念**相似**但**不同**的另一个理论。",
            f"如何**量化**上文提及的现象**对**特定工业的影响?"
        ]
        return mock_questions
        
    def _generate_and_score_queries(self) -> Optional[str]:
        """集成 A 和 B 策略，生成、评分并筛选出最合适的下一查询。"""
        candidates = self._generate_global_queries() + self._generate_local_queries()
        
        # 评分机制：奖励稀疏性和新颖性
        scored_queries = []
        for q in candidates:
            # TODO: 替换为实际的稀疏度评分逻辑（例如，查询 q 涉及的概念在 self._storage['covered_concepts'] 中的覆盖度）
            sparsity_score = np.random.uniform(0.5, 1.0)
            scored_queries.append((q, sparsity_score))

        # 按分数降序排列
        sorted_queries = sorted(scored_queries, key=lambda x: x[1], reverse=True)
        
        # 循环检查，直到找到一个通过多样性检查的查询
        for q, score in sorted_queries:
            if self._check_and_add_query(q):
                print(f"  [筛选] 选中查询: {q[:30]}... (Score: {score:.2f})")
                return q
        
        return None  # 未找到足够多样性的查询

    # --------------------------------------------------------
    # III. 探索与文段续写 (执行与生成)
    # --------------------------------------------------------

    def _mock_query_knowledge_base(self, query: str) -> Tuple[str, List[str]]:
        """【MOCK】模拟知识库查询，并返回答案和提取的知识点。"""
        # TODO: 替换为实际知识库/LLM 查询
        mock_answer = f"知识库返回：查询 '{query[:15]}...' 的结果是：主要涉及**结构**、**功能**和**限制**三个方面。"
        mock_concepts = ["结构", "功能", "限制", "核心机制"]
        return mock_answer, mock_concepts

    def _mock_generate_completion(self, context: str, answer: str) -> str:
        """【MOCK】模拟文段续写生成器。"""
        # TODO: 替换为实际的 LLM 文段续写，将答案合理整合进 context
        new_segment = f"\n[续写文段] 紧接着我们利用新获取的知识：'{answer[:60]}...'，这证实了先前的推测。基于这个新的理解，下一个关键的、但尚未解决的问题被引入了... "
        return new_segment

    # --------------------------------------------------------
    # IV. 主循环驱动
    # --------------------------------------------------------

    def run_iteration(self) -> bool:
        """执行一次查询 -> 探索 -> 续写 -> 更新的迭代循环。"""
        print("\n" + "=" * 60)
        print(f"💡 运行迭代，当前上下文长度: {len(self._storage['latest_context'])}")
        
        # 1. 查询生成与筛选
        selected_query = self._generate_and_score_queries()
        
        if not selected_query:
            print("❌ 无法找到足够多样性的查询。探索结束。")
            return False

        self.current_query = selected_query
        
        # 2. 探索 (执行查询)
        answer, extracted_concepts = self._mock_query_knowledge_base(selected_query)
        
        # 3. 文段续写 (内容生成)
        new_segment = self._mock_generate_completion(self._storage['latest_context'], answer)
        
        # 4. 状态更新
        self._update_covered_knowledge(extracted_concepts)
        self._storage['latest_context'] += new_segment
        
        print(f"\n✨ 下一查询已确定: **{self.current_query}**")
        print("\n--- 最新文段摘要 (末尾 300 字符) ---")
        print(self._storage['latest_context'][-300:])
        return True