from .interfaces import Pipeline

class RAGPipeline(Pipeline):
    def __init__(self, llm, query_rewriter, retriever, reranker, summarizer, constructor, cfg, args):
        self.llm = llm
        self.query_rewriter = query_rewriter
        self.retriever = retriever
        self.reranker = reranker
        self.summarizer = summarizer
        self.constructor = constructor
        self.cfg = cfg
        self.args = args
    def run(self, batch_queries: str) -> str:
        if self.args.rewriter:
            queries_rws = self.query_rewriter.rewrite(batch_queries, n_variants=5)

            original_queries = queries_rws["original_query"]
            rewritten_queries_list = queries_rws["rewritten_queries"]
            all_queries_list = queries_rws["all_queries"]
        else:
            original_queries = [[i] for i in batch_queries]
            rewritten_queries_list = [[None]]
            all_queries_list = [[i] for i in batch_queries] # 如果只有一层的话，那么retriever会将这一组重写得到的query当成多组query来处理

        contexts, doc_ids = self.retriever.retrieve(all_queries_list)
        # 返回格式为 List[List[str]]

        if self.args.reranker:
            contexts, doc_ids  = self.reranker.rerank(contexts, doc_ids, batch_queries)
            # 返回格式为 List[List[str]]
        else:
            contexts = [i[:self.cfg.retrieval["top_n"]] for i in contexts]
            doc_ids = [i[:self.cfg.retrieval["top_n"]] for i in doc_ids]
            # 返回格式为 List[List[str]]

        if self.args.summarizer:
            summarized_contexts = self.summarizer.summarize(contexts, original_queries)
        else:
            summarized_contexts = contexts

        prompt = self.constructor.batch_construct(batch_queries, summarized_contexts)
        # print("[Example Prompt]", prompt[0])
        answers, reasons = self.llm.batch_infer(prompt)
        return contexts, doc_ids, prompt, answers, reasons, rewritten_queries_list, summarized_contexts