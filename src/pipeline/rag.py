from src.interfaces import Pipeline
from src.interfaces import LLMManager, QueryRewriter, Retriever, Reranker, Extractor, PromptConstructor, IntentFilter, ResponseFilter
from typing import List

class RAGPipeline(Pipeline):
    def __init__(self, 
                 llm: LLMManager, query_rewriter: QueryRewriter, retriever: Retriever, reranker: Reranker, extractor: Extractor, constructor: PromptConstructor, 
                 intent_filter: IntentFilter, output_filter: ResponseFilter, 
                 cfg, args):
        self.llm = llm
        self.query_rewriter = query_rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.extractor = extractor
        self.constructor = constructor
        self.intent_filter = intent_filter
        self.output_filter = output_filter
        self.cfg = cfg
        self.args = args
    def run(self, batch_queries: List[str]) -> str:

        if self.args.intent_filter:
            filtered_queries = self.intent_filter.check_intent(batch_queries, verbose=False)
            cleaned_batch_queries = [result["clean_prompt"] for result in filtered_queries]
        else:
            cleaned_batch_queries = batch_queries

        if self.args.rewriter:
            queries_rws = self.query_rewriter.rewrite(cleaned_batch_queries, n_variants=5)

            original_queries = queries_rws["original_query"]
            rewritten_queries_list = queries_rws["rewritten_queries"]
            all_queries_list = queries_rws["all_queries"]
        else:
            original_queries = [[i] for i in cleaned_batch_queries]
            rewritten_queries_list = [[None]]
            all_queries_list = [[i] for i in cleaned_batch_queries] # 如果只有一层的话，那么retriever会将这一组重写得到的query当成多组query来处理

        contexts, doc_ids = self.retriever.retrieve(all_queries_list)
        # 返回格式为 List[List[str]]
        # print("Retrieved contexts finished.")

        if self.args.reranker:
            contexts, doc_ids  = self.reranker.rerank(contexts, doc_ids, cleaned_batch_queries)
            # 返回格式为 List[List[str]]
        else:
            contexts = [i[:self.cfg.retrieval["top_n"]] for i in contexts]
            doc_ids = [i[:self.cfg.retrieval["top_n"]] for i in doc_ids]
            # 返回格式为 List[List[str]]

        if self.args.extractor:
            extracted_contexts = self.extractor.extract(contexts, original_queries)
        else:
            extracted_contexts = contexts

        prompt = self.constructor.batch_construct(cleaned_batch_queries, extracted_contexts)
        # print("Prompt construction finished.")

        answers, reasons = self.llm.batch_infer(prompt)
        # print("LLM inference finished.")

        if self.args.output_filter:
            pass

        return cleaned_batch_queries, contexts, doc_ids, prompt, answers, reasons, rewritten_queries_list, extracted_contexts