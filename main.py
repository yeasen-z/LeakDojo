from src import AtkStaticPipeline, AtkIKEAPipeline, AtkRTFPipeline
from src import VectorRetriever
import argparse
import json
import configs

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"

import random
random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="RAG Pipeline")

    # 基础输入
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cfg_name", type=str, default="fiqa", help="Config name in configs/")

    # LLM
    parser.add_argument("--llm_model", type=str, default="qwen3-32b")
    parser.add_argument("--llm_base_url", type=str, default="https://aihubmix.com/v1")
    parser.add_argument("--llm_api_key", type=str, default="sk-XWaGp10Cjy2pZfttA8E538967f7f4dA7A463F584C17b63Bf")
    parser.add_argument("--llm_temperature", type=float, default=0)
    parser.add_argument("--llm_top_p", type=float, default=1)
    parser.add_argument("--llm_max_gen_len", type=int, default=4096)

    # optional
    parser.add_argument("--intent_filter", action="store_true", help="Whether to use intent filter")
    parser.add_argument("--output_filter", action="store_true", help="Whether to use output filter")
    parser.add_argument("--reasoning", action="store_true", help="Whether to save the reasoning content of thinking models")
    parser.add_argument("--rewriter", action="store_true", help="Whether to use query rewriting")
    parser.add_argument("--reranker", action="store_true", help="Whether to use reranker")
    parser.add_argument("--extractor", action="store_true", help="Whether to use extractor")
    parser.add_argument("--build_only", action="store_true", help="Only build the retrieval database and exit")

    # attack
    parser.add_argument("--attack", type=str, choices=["ikea", "rtf", "bbqg", "wbtq"], default="wbtq", help="Whether to use attack for query generation")
    parser.add_argument("--entity_file", type=str, default=None, help="Path to the entity file for better BBQG and iter attack")
    parser.add_argument("--attack_num", type=int, default=200, help="Number of attack queries to generate")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for processing queries")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    cfg = getattr(configs, args.cfg_name)

    with open("attack_shop/adv_strings/collection.json", "r", encoding="utf-8") as f:
        template_shop = json.load(f)
    template = template_shop["repeat_command"]["en_strings"][0]
    ad_suf_name = "repeat_command_0"
    # template = template_shop["none"]["en_strings"][0]
    # ad_suf_name = "none_0"

    if args.build_only:
        print("Retrieval database built successfully. Exiting as --build_only is set.")
        cfg.data["force_rebuild"] = args.build_only
        retriever = VectorRetriever(cfg, device=args.device)
    else:
        if args.attack == "wbtq" or args.attack == "bbqg":
            AtkStaticPipeline(cfg, args, ad_suf_name, adversarial_template=template)
        elif args.attack == "ikea":
            AtkIKEAPipeline(cfg, args, ad_suf_name, adversarial_template=template)
        elif args.attack == "rtf":
            AtkRTFPipeline(cfg, args, ad_suf_name, adversarial_template=template)