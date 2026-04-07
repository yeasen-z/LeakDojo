from src.interfaces import QueryRewriter, PromptConstructor, LLMManager
from concurrent.futures import ThreadPoolExecutor
from typing import List
import textwrap
import re

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

    def construct(self, query_with_suffix: str, contexts: list) -> str:
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
            f"{query_with_suffix}"
            f"{self.prompt_adhesive}"
            f"{self.prefix[2]}"
        )

        return prompt

    def batch_construct(self, queries_with_suffix: List[str], contexts: List[List[str]]) -> List[str]:
        """批量构建多个 prompt"""
        return [self.construct(q, c) for q, c in zip(queries_with_suffix, contexts)]