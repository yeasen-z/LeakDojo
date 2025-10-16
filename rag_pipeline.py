from references import zeng24_config, zeng24_question
from src.retrieval import VRConfig, VectorRetriever, RerankerManager, LLMHybridSummarization
from src.prompts import LLMQueryRewriter, SimplePromptConstructor
from config import database_desc
from src.llm import OpenAILLM
import argparse
import os
import json
from src.utils import get_retrieval_info, get_data_chunks, get_data_chunks_by_params, get_llm_output_file


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Pipeline")

    # 基础输入
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--cfg_name", type=str, default="Zeng24fiqa")

    # retrieval
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild retrieval database")


    # LLM
    parser.add_argument("--llm_model", type=str, default="./Models/Qwen3-14B")
    parser.add_argument("--llm_base_url", type=str, default="http://localhost:22999/v1")
    parser.add_argument("--llm_api_key", type=str, default="EMPTY")
    parser.add_argument("--llm_temperature", type=float, default=0)
    parser.add_argument("--llm_top_p", type=float, default=1)
    parser.add_argument("--llm_max_gen_len", type=int, default=4096)

    # optional
    parser.add_argument("--reasoning", action="store_true", help="Whether to save the reasoning content of thinking models")
    parser.add_argument("--rewriter", action="store_true", help="Whether to use query rewriting")
    parser.add_argument("--reranker", action="store_true", help="Whether to use reranker")
    parser.add_argument("--summarizer", action="store_true", help="Whether to use summarization")

    return parser.parse_args()

def setup(cfg, args):
    # Retrieval
    retrieval_name, retrieval_store_path = get_retrieval_info(cfg)

    retriever_config = VRConfig()
    retriever_config.update_4m_dict({
        "data": {
                "retrieval_name": retrieval_name,
                "retrieval_store_path": retrieval_store_path,
                "force_rebuild": args.force_rebuild,
                "datastorage_tool": "chroma",
                "data_dir_list": cfg.datastorage.raw_data_dir,
            },
        "retrieval": {
                "method": cfg.retrieval.method,
                "top_k": cfg.retrieval.params.get("k", 15),
                "fetch_k": cfg.retrieval.params.get("fetch_k", 60),
                "score_threshold": cfg.retrieval.params.get("score_threshold", 0.75),
                "top_n": cfg.retrieval.params.get("n", 10)
            },
        "embed": {
                "provider": cfg.embedding.provider,
                "model_dir": cfg.embedding.model_dir
            }
    })

    # 初始化
    llm = OpenAILLM(model = args.llm_model, 
                    base_url = args.llm_base_url, 
                    api_key = args.llm_api_key, 
                    reasoning= args.reasoning,
                    temperature=args.llm_temperature,
                    top_p=args.llm_top_p,
                    max_gen_len=args.llm_max_gen_len,
                    max_workers=50)

    query_rewriter = LLMQueryRewriter(llm,database_desc)

    retriever = VectorRetriever(retriever_config, device=args.device)
    if cfg.retrieval.rerank:
        reranker = RerankerManager(reranker_model=cfg.retrieval.rerank, top_n=retriever_config.retrieval['top_n'], device=args.device)
    else:
        reranker = None

    if args.rewriter and not args.reranker:
        print("[NOTING] Query rewriting is enabled but Reranker is disabled. It's recommended to use Query Rewriter with Reranker for better performance.")
    
    summarizer = LLMHybridSummarization(llm, embed_provider=cfg.embedding.provider, embed_model_dir=cfg.embedding.model_dir, device='cuda:1')    
    prompt_constructor = SimplePromptConstructor()

    return llm, query_rewriter, retriever, reranker, summarizer, prompt_constructor


def main():
    args = parse_args()

    cfg = getattr(zeng24_config, args.cfg_name)() if hasattr(zeng24_config, args.cfg_name) else None

    llm, query_rewriter, retriever, reranker, summarizer, prompt_constructor = setup(cfg, args)

    queries = ["What is Dividend Growth?", "What is CPI?"]

    # 暂存数据结果
    save_helper = {
        "queries": [],
        "rewritten_queries": [],
        "contexts": [],
        "doc_ids": [],
        "sum_contexts": [],
        "prompts": [],
        "answers": [],
        "reasons": []
    }

    for query in queries:
        print(f"Processing query: {query}")

        if args.rewriter:
            queries_rws = query_rewriter.rewrite([query], n_variants=5)

            original_queries = queries_rws["original_query"]
            rewritten_queries_list = queries_rws["rewritten_queries"]
            all_queries_list = queries_rws["all_queries"]

            save_helper["rewritten_queries"].append(rewritten_queries_list)
        else:
            original_queries = [query]
            rewritten_queries_list = [[None]]
            all_queries_list = [[query]] # 如果只有一层的话，那么retriever会将这一组重写得到的query当成多组query来处理

        contexts, doc_ids = retriever.retrieve(all_queries_list)
        # 返回格式为 List[List[str]]

        if args.reranker:
            contexts, doc_ids  = reranker.rerank(contexts, doc_ids, query)
            # 返回格式为 List[List[str]]
        else:
            contexts = [i[:cfg.retrieval.params.get("n", 10)] for i in contexts]
            doc_ids = [i[:cfg.retrieval.params.get("n", 10)] for i in doc_ids]
            # 返回格式为 List[List[str]]

        if args.summarizer:
            summarized_contexts = summarizer.summarize(contexts, original_queries)
            save_helper["sum_contexts"].append(summarized_contexts)
        else:
            summarized_contexts = contexts

        print(f"Generating the prompt of {query}...")
        prompt = prompt_constructor.batch_construct([query], summarized_contexts)
        answers, reasons = llm.batch_infer(prompt)

        save_helper["queries"].append(original_queries)
        save_helper["contexts"].append(contexts)
        save_helper["doc_ids"].append(doc_ids)
        save_helper["prompts"].append(prompt)
        save_helper["answers"].append(answers)
        save_helper["reasons"].append(reasons)

    # output_dir = get_llm_output_file(cfg)
    # os.makedirs(output_dir, exist_ok=True)
    # save_path = os.path.join(output_dir, get_llm_output_file(cfg))
    # with open(save_path, "w", encoding="utf-8") as f_a:
    #     json.dump(save_helper, f_a, ensure_ascii=False, indent=2)

    output_dir = cfg.expconfig.output_dir
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, "all_save_helper.json")
    print(f"Saving results to {save_path}")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_helper, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()