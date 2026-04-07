from src.components import OpenAILLM, LLMQueryRewriter, VectorRetriever, RerankerManager, LLMHybridExtractor, SimplePromptConstructor
from src.components import LLMIntentFilter, RougeLResponseFilter

def chunked(iterable, batch_size):
    """把列表按 batch_size 分块"""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]
        
def setup(cfg, args):
    # 初始化
    llm = OpenAILLM(model = args.llm_model, 
                    base_url = args.llm_base_url, 
                    api_key = args.llm_api_key, 
                    reasoning = args.reasoning,
                    temperature = args.llm_temperature,
                    top_p = args.llm_top_p,
                    max_gen_len = args.llm_max_gen_len,
                    max_workers=50)
    
    llm_tool = OpenAILLM(model = cfg.tool_llm["model"], 
                    base_url = cfg.tool_llm["base_url"], 
                    api_key = cfg.tool_llm["api_key"], 
                    reasoning = cfg.tool_llm["reasoning"],
                    temperature = cfg.tool_llm["temperature"],
                    top_p = cfg.tool_llm["top_p"],
                    max_gen_len = 1024,
                    max_workers = 50)

    query_rewriter = LLMQueryRewriter(llm_tool, cfg.data["description"])

    retriever = VectorRetriever(cfg, device=args.device)
    if cfg.reranker["model"]:
        reranker = RerankerManager(reranker_config=cfg.reranker, top_n=cfg.retrieval['top_n'], device=args.device)
    else:
        reranker = None

    if args.rewriter and not args.reranker:
        print("[NOTING] Query rewriting is enabled but Reranker is disabled. It's recommended to use Query Rewriter with Reranker for better performance.")

    extractor = LLMHybridExtractor(llm_tool, extract_config=cfg.extractor, device=args.device)
    constructor = SimplePromptConstructor()

    intent_filter, output_filter = LLMIntentFilter(llm_tool=llm_tool), RougeLResponseFilter()
    
    return llm, llm_tool, intent_filter, output_filter, query_rewriter, retriever, reranker, extractor, constructor