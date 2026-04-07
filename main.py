from src import AtkStaticPipeline, AtkIKEAPipeline, AtkRTFPipeline, AtkPoRPipeline, AtkDGEAPipeline
from src import VectorRetriever
import argparse
import glob
import json
import configs

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import random
random.seed(42)

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description="RAG Pipeline")

    # 基础输入
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--cfg_name", type=str, default="fiqa", help="Config name in configs/")

    # LLM
    parser.add_argument("--llm_model", type=str, default="YOUR_MODEL_NAME")
    parser.add_argument("--llm_base_url", type=str, default="YOUR_BASE_URL")
    parser.add_argument("--llm_api_key", type=str, default="EMPTY")
    parser.add_argument("--llm_temperature", type=float, default=0)
    parser.add_argument("--llm_top_p", type=float, default=1)
    parser.add_argument("--llm_max_gen_len", type=int, default=2048)

    # optional
    parser.add_argument("--intent_filter", action="store_true", help="Whether to use intent filter")
    parser.add_argument("--output_filter", action="store_true", help="Whether to use output filter")
    parser.add_argument("--reasoning", action="store_true", help="Whether to save the reasoning content of thinking models")
    parser.add_argument("--rewriter", action="store_true", help="Whether to use query rewriting")
    parser.add_argument("--reranker", action="store_true", help="Whether to use reranker")
    parser.add_argument("--extractor", action="store_true", help="Whether to use extractor")
    parser.add_argument("--build_only", action="store_true", help="Only build the retrieval database and exit")

    # attack
    parser.add_argument("--attack", type=str, choices=["ikea", "rtf", "pide", "wbtq", "por", "dgea", "tgtb"], default="wbtq", help="Whether to use attack for query generation")
    parser.add_argument("--lma_method", type=str, default=None, help="LMA method name from leakdojo.json to override adversarial template (e.g. reranker_role_play)")
    parser.add_argument("--entity_file", type=str, default=None, help="Path to the entity file for better iter attack")
    parser.add_argument("--attack_num", type=int, default=200, help="Number of attack queries to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing queries")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    cfg = getattr(configs, args.cfg_name)
    cfg.tool_llm["base_url"] = args.llm_base_url
    cfg.tool_llm["api_key"] = args.llm_api_key

    template_shop = {}
    for fp in sorted(glob.glob("attack_shop/adv_strings/*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            template_shop.update(json.load(f))

    # 全局 LMA 模板覆盖
    lma_template = None
    if args.lma_method:
        lma_pool = {k: v for k, v in template_shop.items() if isinstance(v, dict) and v.get("belongs") == ["LMA"]}
        if args.lma_method in lma_pool:
            lma_template = lma_pool[args.lma_method]["en_strings"][0]

    if args.build_only:
        print("Retrieval database built successfully. Exiting as --build_only is set.")
        cfg.data["force_rebuild"] = args.build_only
        retriever = VectorRetriever(cfg, device=args.device)
    else:
        if args.attack == "wbtq" or \
            args.attack == "pide" or \
                args.attack == "tgtb":
            if args.attack == "tgtb":
                template = template_shop["tgtb"]["en_strings"][0]
                ad_suf_name = "tgtb"
            elif args.attack == "pide":
                template = template_shop["pide"]["en_strings"][0]
                ad_suf_name = "pide"
            else:
                template = template_shop["repeat_command"]["en_strings"][0]
                ad_suf_name = "repeat_command_0"
            if lma_template:
                template = lma_template
                ad_suf_name = f"lma_{args.lma_method}"
            AtkStaticPipeline(cfg, args, ad_suf_name, adversarial_template=template)
        elif args.attack == "ikea":
            template = template_shop["repeat_command"]["en_strings"][0]
            ad_suf_name = "repeat_command_0"
            if lma_template:
                template = lma_template
                ad_suf_name = f"lma_{args.lma_method}"
            AtkIKEAPipeline(cfg, args, ad_suf_name, adversarial_template=template)
        elif args.attack == "rtf":
            template = template_shop["rag_thief"]["en_strings"][0]
            ad_suf_name = "rtf_attack"
            if lma_template:
                template = lma_template
                ad_suf_name = f"lma_{args.lma_method}"
            AtkRTFPipeline(cfg, args, ad_suf_name, adversarial_template=template)
        elif args.attack == "por":
            AtkPoRPipeline(cfg, args, lma_template=lma_template)
        elif args.attack == "dgea":
            AtkDGEAPipeline(cfg, args)
