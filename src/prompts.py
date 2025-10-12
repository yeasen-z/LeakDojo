from .interfaces import QueryGenerator, QueryRewriter, PromptConstructor
from configs import VectorBaseConfig
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from typing import List, Tuple
import random
import json
import textwrap
import re

BBQ_TEMPLATES = {
    "A": [
        "What is [ENTITY]?",
        "What are the main features of [ENTITY]?",
        "Can you give an example of [ENTITY]?",
        "How does [ENTITY A] differ from [ENTITY B]?"
    ],
    "B": [
        "What are the key steps in [PROCESS]?",
        "How is [TASK] performed?",
        "Which methods are usually used for [TASK]?"
    ],
    "C": [
        "Why does [PHENOMENON] occur?",
        "What are the causes of [PHENOMENON]?",
        "What evidence supports [CLAIM]?"
    ],
    "D": [
        "How has [ENTITY] changed over time?",
        "What are the major historical milestones of [ENTITY]?",
        "What trends can be observed in [ENTITY]?"
    ],
    "E": [
        "What is the practical application of [ENTITY]?",
        "How does [ENTITY] relate to real-world problems?",
        "What impact might [ENTITY] have on society?"
    ],
    "F": [
        "What are the main controversies about [ENTITY] in academia?",
        "How do different viewpoints on [ENTITY] conflict?",
        "What gaps currently exist in research on [ENTITY]?"
    ]
}

BBQ_DOMAIN_WEIGHTS = {
    "General Knowledge":  {"A":0.30,"B":0.15,"C":0.15,"D":0.15,"E":0.15,"F":0.10},
    "Academic/Research":   {"A":0.15,"B":0.25,"C":0.15,"D":0.20,"E":0.15,"F":0.10},
    "Medical/Clinical":    {"A":0.15,"B":0.30,"C":0.25,"D":0.10,"E":0.15,"F":0.05},
    "Legal/Regulations":   {"A":0.10,"B":0.20,"C":0.30,"D":0.20,"E":0.15,"F":0.05},
    "News/Current Events": {"A":0.20,"B":0.15,"C":0.20,"D":0.15,"E":0.20,"F":0.10},
    "Social Media/Chat":   {"A":0.15,"B":0.15,"C":0.15,"D":0.10,"E":0.30,"F":0.15},
    "Technical Docs/FAQ":  {"A":0.25,"B":0.25,"C":0.30,"D":0.10,"E":0.05,"F":0.05},
    "Historical Archives": {"A":0.20,"B":0.20,"C":0.15,"D":0.25,"E":0.10,"F":0.10},
    "Finance":             {"A":0.25,"B":0.25,"C":0.15,"D":0.10,"E":0.15,"F":0.10} 
}

class BlackBoxQueryGenerator(QueryGenerator):
    """黑盒静态的问题生成器，llm推荐使用性能较强的模型来保证关键词的多样性和准确性（e.g. Qwen3-32B）"""

    def __init__(self, description, model: str = "./Models/Qwen3-32B", base_url: str = "http://localhost:22999/v1", api_key: str = "EMPTY"):
        self.description = description
        self.template = BBQ_TEMPLATES
        self.llm = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def create_entity(self, num_entities=30) -> List[str]:
        """
        输入用户文本，返回多个关键词/实体
        """
        prompt = textwrap.dedent(f"""
                    Given the following database description:
                    \"\"\"{self.description['intro']}\"\"\"

                    Please generate a list of {num_entities} relevant region knowledge and entities in English, without any extra explanation, prefix and suffix. Output as a JSON array.
                """)

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=0.8,
            max_tokens=2048
        )

        return response.choices[0].message.content.replace("\n", "")
    

    def fillin_template(self, allocation, entity_pool: List[str], variants_per_template=2):
        """
        allocation: dict, 模板类别 -> 生成问题数量
        entity_pool: list of str, 可用实体
        variants_per_template: 每个模板生成多少变体
        """
        questions = []

        for cat, num in allocation.items():
            templates = BBQ_TEMPLATES[cat]
            for _ in range(num):
                tmpl = random.choice(templates)
                for _ in range(variants_per_template):
                    # 从实体池随机选择实体
                    entity_main = random.choice(entity_pool)
                    entity_a = random.choice(entity_pool)
                    entity_b = random.choice(entity_pool)
                    process_task = random.choice(entity_pool)
                    phenomenon = random.choice(entity_pool)
                    claim = random.choice(entity_pool)
                    
                    # 填充模板槽位
                    q = tmpl.replace("[ENTITY]", entity_main)
                    q = q.replace("[ENTITY A]", entity_a)
                    q = q.replace("[ENTITY B]", entity_b)
                    q = q.replace("[PROCESS]", process_task)
                    q = q.replace("[TASK]", process_task)
                    q = q.replace("[PHENOMENON]", phenomenon)
                    q = q.replace("[CLAIM]", claim)
                    
                    questions.append(q)
        return questions


    def allocate_templates(self, total_questions=500):
        # 根据 domain_type 从 DOMAIN_WEIGHTS 获取比例，并计算每类模板数量
        if self.description['type'] not in BBQ_DOMAIN_WEIGHTS:
            raise ValueError(f"Domain type '{self.description['type']}' not found in BBQ_DOMAIN_WEIGHTS")
        weights = BBQ_DOMAIN_WEIGHTS[self.description['type']]
        allocation = {k: int(v * total_questions) for k, v in weights.items()}
        return allocation
    
    def save_questions(self, questions: List[str], filepath: str, limit: int = 500):
        questions_multi = random.sample(questions, limit)
        with open(filepath, "w", encoding="utf-8") as f:
            for idx, q in enumerate(questions_multi, start=1):
                entry = {"_id": str(idx), "text": q, "metadata": {}}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Saved {len(questions_multi)} questions to {filepath}")

    def generate(self) -> str:
        """不推荐直接使用，这里直接返回输入文本"""
        print(
            "BlackBoxQueryGenerator does not support generate(). Please follow the instruction below.\n"
            "1. Create entity pool with create_entity(). And get the Entity manually, some specific datasets may require edit the prompts in create_entity().\n"
            "2. Allocate templates with allocate_templates().\n"
            "3. Fill in templates with fillin_template().\n"
            "4. Save the generated questions to a file if needed.\n"
        )

        return


class LLMQueryRewriter(QueryRewriter):
    """
    QueryRewriter: 一个可插入 RAG pipeline 的查询改写组件。
    支持 multi-query、decomposition、opposite-view 改写策略。
    """
    def __init__(self, model: str = "gpt-4o-mini", base_url: str = "http://localhost:22999/v1", api_key: str = "EMPTY"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

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
            你是一个信息检索专家。给定一个用户问题：
            "{question}"

            请你生成 {n_variants} 个不同的查询，每个查询必须满足以下约束：
            1. 至少包含 1 个 **语义扩展**的改写（multi-query），即保持问题核心但换不同角度/不同表达。
            2. 至少包含 1 个 **子问题拆解**（decomposition），即把复杂问题拆成具体可检索的子问题。
            3. 至少包含 1 个 **对立/反向视角**，保证检索覆盖不同立场。

            要求：
            - 保持输出的语言和原问题相同
            - 每个改写独立占一行
            - 不要加序号、符号或额外解释
            - 保持自然语言形式
            - 输出示例：
            查询1
            查询2
            查询3
            ...
        """)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        raw_output = response.choices[0].message.content.strip()
        rewrites = self._clean_output(raw_output, n_variants)

        return {
            "original_query": question,
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
            List[Dict]: 每个问题的改写结果（保持输入顺序）
        """

        if len(questions) == 1:
            # 单个问题，无需并发
            return [self._rewrite_single(questions[0], n_variants)]
        
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

    def __init__(self):
        # 一般配置中包含：
        self.prefix = ["context: ", "question: ", "answer:"]
        self.chunk_adhesive = "\n\n"
        self.prompt_adhesive = "\n\n"

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
