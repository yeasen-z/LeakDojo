from .interfaces import QueryGenerator, QueryRewriter, PromptConstructor
from configs import VectorBaseConfig
from openai import OpenAI
from typing import List, Tuple
import random
import json
import textwrap


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
