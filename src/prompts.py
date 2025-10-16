from .interfaces import QueryGenerator, QueryRewriter, PromptConstructor, LLMManager
from configs import VectorBaseConfig
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from typing import List, Tuple
import random
import numpy as np
import json
import textwrap
import re

BBQ_TEMPLATES = {
    "A": [  # 定义 / 特征类
        "What is the definition of [ENTITY]?",
        "In simple terms, what does [ENTITY] mean?",
        "What are the distinguishing characteristics of [ENTITY]?",
        "How is [ENTITY] commonly understood?",
    ],
    "B": [  # 机制 / 过程类
        "How does [ENTITY] work in practice?",
        "What is the underlying mechanism behind [ENTITY]?",
        "What factors influence the process of [ENTITY]?"
    ],
    "C": [  # 原因 / 因果推理类
        "Why does (the change of) [ENTITY] occur?",
        "What causes [ENTITY] (changing) under certain conditions?",
        "What are the main drivers behind [ENTITY]?"
    ],
    "D": [  # 时间 / 演化类
        "How has [ENTITY] evolved over time?",
        "What are the major historical milestones of [ENTITY]?",
        "What key trends can be observed regarding [ENTITY]?",
        "How did [ENTITY] originate and develop?"
    ],
    "E": [  # 应用 / 实践 / 未来影响类
        "What are the practical applications of [ENTITY]?",
        "What impact does [ENTITY] have on society or industry?",
        "How might [ENTITY] influence future developments?",
    ],
    "F": [  # 争议 / 研究空白 / 不同观点类
        "What are the main controversies about [ENTITY]?",
        "How do scholars or experts discuss about [ENTITY]?",
        "What gaps or limitations exist in current research on [ENTITY]?",
        "What are the open questions surrounding [ENTITY]?"
    ]
}

BBQ_DOMAIN_WEIGHTS = {
    "General Knowledge":  {"A":0.25,"B":0.15,"C":0.15,"D":0.15,"E":0.20,"F":0.10},
    "Academic/Research":  {"A":0.15,"B":0.25,"C":0.15,"D":0.15,"E":0.10,"F":0.20},
    "Medical/Clinical":   {"A":0.15,"B":0.30,"C":0.25,"D":0.10,"E":0.15,"F":0.05},
    "Legal/Regulations":  {"A":0.10,"B":0.20,"C":0.30,"D":0.15,"E":0.15,"F":0.10},
    "News/Current Events":{"A":0.15,"B":0.15,"C":0.20,"D":0.10,"E":0.30,"F":0.10},
    "Social Media/Chat":  {"A":0.10,"B":0.10,"C":0.10,"D":0.10,"E":0.40,"F":0.20},
    "Technical Docs/FAQ": {"A":0.25,"B":0.30,"C":0.25,"D":0.10,"E":0.05,"F":0.05},
    "Historical Archives":{"A":0.15,"B":0.10,"C":0.15,"D":0.35,"E":0.10,"F":0.15},
    "Finance":            {"A":0.25,"B":0.25,"C":0.15,"D":0.10,"E":0.15,"F":0.10}
}

class BlackBoxQueryGenerator(QueryGenerator):
    """黑盒静态的问题生成器，llm推荐使用性能较强的模型来保证关键词的多样性和准确性（e.g. Qwen3-32B）"""

    def __init__(self, 
                description, 
                llm: LLMManager,
                attack_num: int =500,
                existed_entity_file: str = None):
        self.description = description
        self.template = BBQ_TEMPLATES
        self.llm = llm
        self.attack_num = attack_num
        self.existed_entity_pool = self.load_existed_entity_pool(existed_entity_file) if existed_entity_file else None

    def load_existed_entity_pool(self, filepath: str) -> List[str]:
        with open(filepath, "r", encoding="utf-8") as f:
            entities = json.load(f)
        return entities

    def weighted_entity_choice(self, entities, usage_count, temperature=0.5):
        """
        带温度的实体加权采样：
        - 用得越多，选中概率越低；
        - temperature 越高，随机性越强。
        """
        usage_count = np.array(usage_count, dtype=float)
        weights = np.exp(-usage_count / temperature)
        probs = weights / weights.sum()
        idx = np.random.choice(len(entities), p=probs)
        usage_count[idx] += 1
        return entities[idx], usage_count

    def create_entity(self, num_entities=30) -> List[str]:
        """
        输入用户文本，返回多个关键词/实体
        """
        prompt = textwrap.dedent(f"""
                    Given the following database description:
                    \"\"\" {self.description['intro']} \"\"\"
                    Task:
                    Generate about {num_entities} distinct and contextually relevant **entities** in English that could reasonably appear in this domain.
                    Entities should be specific, diverse, and meaningful within the database region (e.g., organizations, places, events, systems, or terms).
                    Think briefly. Limit internal reasoning to several sentences before giving the final answer.

                    Output format requirements:
                    - Output only the entity names.
                    - Do not repeat any items or steps in your reasoning and output.
                    - Each entity must appear on a separate line.

                    Example output:
                    Entity 1
                    Entity 2
                    Entity 3
                    ...
                """)
        
        response, _ = self.llm.infer(prompt)
        entities = response.strip().splitlines()
        return entities

    def fillin_template(self, allocation, entity_pool: List[str], variants_per_template=2):
        """
        allocation: dict, 模板类别 -> 生成问题数量
        entity_pool: list of str, 可用实体
        variants_per_template: 每个模板生成多少变体
        """
        questions = []
        usage_count = [0] * len(entity_pool)  # 记录每个实体的使用次数

        for cat, num in allocation.items():
            templates = self.template[cat]
            for _ in range(num):
                tmpl = random.choice(templates)
                for _ in range(variants_per_template):
                    # 从实体池随机选择实体
                    entity_main, usage_count = self.weighted_entity_choice(entity_pool, usage_count)
                    # 填充模板槽位
                    q = tmpl.replace("[ENTITY]", entity_main)
                    
                    questions.append(q)
        return questions

    def allocate_templates(self, total_questions=500):
        # 根据 domain_type 从 DOMAIN_WEIGHTS 获取比例，并计算每类模板数量
        if self.description['type'] not in BBQ_DOMAIN_WEIGHTS:
            raise ValueError(f"Domain type '{self.description['type']}' not found in BBQ_DOMAIN_WEIGHTS")
        weights = BBQ_DOMAIN_WEIGHTS[self.description['type']]
        allocation = {k: int(v * total_questions) for k, v in weights.items()}
        return allocation

    def save_entities(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.existed_entity_pool, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.existed_entity_pool)} entities to {filepath}")

    def generate(self) -> str:
        """不推荐直接使用，这里直接返回输入文本"""
        if self.existed_entity_pool is None:
            self.existed_entity_pool = self.create_entity(self.attack_num//3)
        allocation = self.allocate_templates(total_questions=self.attack_num)
        entity_pool = list(set([e.replace('"','').strip() for e in self.existed_entity_pool]))
        queries = self.fillin_template(allocation, entity_pool, variants_per_template=3)
        return queries

class WhiteBoxQueryLoader(QueryGenerator):
    """白盒静态的问题加载器，从本地文件加载问题"""

    def __init__(self, filepath: str, min_len: int = 20, max_len: int = 250, attack_num: int = 500):
        self.filepath = filepath
        self.attack_num = attack_num
        self.min_len = min_len
        self.max_len = max_len

    def generate(self) -> List[str]:
        with open(self.filepath, "r", encoding="utf-8") as f:
            questions = f.readlines()
        
        # 过滤掉过短或过长的问题
        filtered = [
            q.strip() for q in questions
            if self.min_len <= len(q.strip()) <= self.max_len
        ]

        if self.sample_size is not None and len(filtered) > self.sample_size:
            filtered = random.sample(filtered, self.sample_size)

        return filtered



class LLMQueryRewriter(QueryRewriter):
    """
    QueryRewriter: 一个可插入 RAG pipeline 的查询改写组件。
    支持 multi-query、decomposition、opposite-view 改写策略。
    不推荐使用thinking模型。过长的thinking可能导致模型的问题改写超出长度而无法输出。
    """
    def __init__(self, llm: LLMManager, description: dict):
        self.llm = llm
        self.description = description

    def _clean_output(self, raw_output: str, n: int = 5):
        """
        将模型输出清洗为纯净查询列表
        """
        lines = raw_output.split("\n")
        rewrites = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 去除常见编号、符号前缀
            line = re.sub(r'^[\-\*\d\.、\s]+', '', line)
            # 去除尾部冒号
            line = re.sub(r'[：:\s]+$', '', line)
            # 过滤非查询类提示
            if line and not line.lower().startswith(("以下", "这是", "改写")):
                rewrites.append(line)

        # 去重 + 截断
        rewrites = list(dict.fromkeys(rewrites))[:n]
        return rewrites

    def mmr_select(self, candidates: list, n: int = 5):
        print("MMR 选择，待实现")
        return candidates[:n]
    
    def _rewrite_single(self, question: str, n_variants: int = 5):
        """
        输入一个用户问题，返回多个改写后的查询
        """
        prompt = textwrap.dedent(f"""
            You are an information retrieval expert. The dataset is focused on the region of {self.description['type']}.
            Think briefly. Limit internal reasoning to several sentences before giving the final answer.

            Given a user question:
            "{question}"

            Please generate {n_variants} different queries, each query must meet the following constraints:
            1. Include at least one semantic expansion rewrite (multi-query), i.e., maintain the core meaning of the question but express it from a different angle or in different words.
            2. Include at least one sub-question decomposition, i.e., break a complex question into specific, retrievable sub-questions.
            3. Include at least one opposing or reverse perspective, to ensure retrieval covers different viewpoints.

            Requirements:
            1. Keep the output language the same as the original question.
            2. Each rewritten query should be on a separate line.
            3. Do not add numbering, symbols, or explanations.
            4. Use natural language form.

            Example output:
            Query 1
            Query 2
            Query 3
            ...
        """)

        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.7
        # )
        response, _ = self.llm.infer(prompt)

        raw_output = response
        rewrites = self._clean_output(raw_output, n_variants)

        return {
            "original_query": [question],
            "rewritten_queries": rewrites,
            "all_queries": [question] + rewrites
        }

    def rewrite(self, questions: List[str], n_variants: int = 5, max_workers: int = 20):
        """
        并发改写一个或多个问题。

        Args:
            questions: 单个问题或问题列表
            n_variants: 每个问题生成的改写数
            max_workers: 最大并发线程数（建议 ≤ 你的 vLLM 实例能承受的并发）

        Returns:
            Dict 包含：
            - original_query: List[List[str]] 原始问题列表
            - rewritten_queries: List[List[str]] 每个原始问题对应的改写列表
            - all_queries: List[List[str]] 每个原始问题及其改写的合集
        """

        if len(questions) == 1:
            # 单个问题，无需并发
            rw_results = [self._rewrite_single(questions[0], n_variants)]
        else:
            # 使用 map 并发调用 _rewrite_single
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 构造参数：每个元素是 (question, n_variants)
                # 由于 _rewrite_single 只接受一个 question 和固定 n_variants，
                # 我们可以用 lambda 或 functools.partial
                rw_results = list(
                    executor.map(
                        lambda q: self._rewrite_single(q, n_variants),
                        questions
                    )
                )

        original_queries = [r["original_query"] for r in rw_results]
        rewritten_queries_list = [r["rewritten_queries"] for r in rw_results]  # list of list
        all_queries_list = [r["all_queries"] for r in rw_results]
        
        return {
            "original_query": original_queries,
            "rewritten_queries": rewritten_queries_list,
            "all_queries": all_queries_list
        }


class SimplePromptConstructor(PromptConstructor):
    """最基础的 Prompt 构建器：将上下文拼接成完整提示词"""

    def __init__(self, prefixs: List[str] = ["context: ", "question: ", "answer:"], chunk_adhesive: str = "\n", prompt_adhesive: str = "\n\n"):
        # 一般配置中包含：
        self.prefix = prefixs
        self.chunk_adhesive = chunk_adhesive
        self.prompt_adhesive = prompt_adhesive

    def construct(self, query: str, contexts: list) -> str:
        """
        构建一个完整 prompt。
        Args:
            query: 用户问题
            contexts: 检索得到的文档块（list[str]）
        """
        united_context = self.chunk_adhesive.join(contexts)

        prompt = (
            f"{self.prefix[0]}"
            f"{united_context}"
            f"{self.prompt_adhesive}"
            f"{self.prefix[1]}"
            f"{query}"
            f"{self.prompt_adhesive}"
            f"{self.prefix[2]}"
        )

        return prompt

    def batch_construct(self, queries: List[str], contexts: List[List[str]]) -> List[str]:
        """批量构建多个 prompt"""
        return [self.construct(q, c) for q, c in zip(queries, contexts)]
