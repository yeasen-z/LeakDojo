from abc import ABC, abstractmethod


class QueryGenerator(ABC):
    @abstractmethod
    def generate(self) -> str:
        """问题生成器：生成锚点问题"""
        pass

class QueryRewriter(ABC):
    @abstractmethod
    def rewrite(self, query: str) -> list:
        """查询改写器：输入 query，输出多个改写后的 query 列表"""
        pass

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> list:
        """检索器：输入 query，输出候选文档列表"""
        pass

class Reranker(ABC):
    @abstractmethod
    def rerank(self, docs: list, docs_id: list, query: str) -> list:
        """可选：对检索到的文档重新排序"""
        pass

class Summarizer(ABC):
    @abstractmethod
    def summarize(self, chunks: list) -> str:
        """可选：对长文档进行分块摘要"""
        pass

class PromptConstructor(ABC):
    @abstractmethod
    def construct(self, query: str, contexts: list) -> str:
        """Prompt 构造器：将 query 和得到的上下文拼接成最终的 prompt"""
        pass

class LLM(ABC):
    @abstractmethod
    def infer(self, prompt: str) -> str:
        """大模型：输入 prompt，输出生成结果"""
        pass

class AttackEvaluator(ABC):
    @abstractmethod
    def evaluate(self, prediction: str, reference: str) -> float:
        """可选：评估模块，比如计算准确率、BLEU、ASR"""
        pass

class Pipeline(ABC):
    @abstractmethod
    def run(self, input_text: str) -> str:
        """流水线：将各组件串联起来，输入用户文本，输出最终结果"""
        pass