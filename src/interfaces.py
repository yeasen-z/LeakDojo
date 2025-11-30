from abc import ABC, abstractmethod
from typing import Union, List


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

class Extractor(ABC):
    @abstractmethod
    def extract(self, chunks: list) -> str:
        """可选：对长文档进行分块摘要"""
        pass

class PromptConstructor(ABC):
    @abstractmethod
    def construct(self, query: str, contexts: list) -> str:
        """Prompt 构造器：将 query 和得到的上下文拼接成最终的 prompt"""
        pass

class LLMManager(ABC):
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

class IntentFilter(ABC):
    @abstractmethod
    def check_intent(self, user_input: Union[str, List[str]], verbose: bool = False) -> dict:
        """意图过滤器：判断用户输入的意图是否安全"""
        pass

class ResponseFilter(ABC):
    @abstractmethod
    def check_output(self, llm_output: Union[str, List[str]]) -> str:
        """输出过滤器：对 LLM 输出进行过滤和修正"""
        pass