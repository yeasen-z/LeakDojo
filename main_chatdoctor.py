from references import zeng24_config, zeng24_question
from rag_components import vector_retriever, get_prompts, vector_retrieved_contexts, run_llm, vector_embed_model
from tools import load_saved_data, eva_pub_pri_hitnum, eva_pii_hitnum, eva_repeat_context, eva_rouge, eva_bleu, eva_embedding_similarity
import torch
import os
import argparse

def main():
    '''
    1. 配置文件，每个方法单独实现
    2. 问题生成，每个方法单独实现
    3. 根据配置文件生成retriever
    4. 根据retriever和问题生成prompt，并对应保存 contexts，question，prompts
    5. prompts进行summarize
    6. 根据prompts生成answers，并保存
    7. 评估模块，单独实现
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["inference", "evaluation", "build_data"], required=True,
        help="Choose whether to run inference or evaluation"
    )
    args = parser.parse_args()

    # Run inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg = zeng24_config.Zeng24ChatDoctor()    
    
    if args.mode == "inference":
        # 保存文件的目录
        if not os.path.exists(cfg.expconfig.output_dir):
            os.makedirs(cfg.expconfig.output_dir,exist_ok=True)

        questions = zeng24_question.get_question(**zeng24_question.zeng24_chatdoctor_q)
            
        retriver = vector_retriever(cfg, retrival_database_batch_size=512, device=device, force_rebuild=False)

        prompts = get_prompts(cfg, retriver, questions, device=device)

        answers = run_llm(cfg, prompts)
    
    elif args.mode == "build_data":
        # 保存文件的目录
        if not os.path.exists(cfg.expconfig.output_dir):
            os.makedirs(cfg.expconfig.output_dir,exist_ok=True)

        # 构建向量数据库
        _ = vector_retriever(cfg, retrival_database_batch_size=512, device=device, force_rebuild=True)

    elif args.mode == "evaluation":
        # Run evaluation
        doc_ids, outputs, contexts, question = load_saved_data(cfg)
        question_num = len(question)
        print(f"Total question num: {question_num}")

        num_effective_prompt, avg_effective_length, num_extract_context = eva_repeat_context(doc_ids, outputs, contexts)
        print(f"Total effective prompt num: {num_effective_prompt},  Average extracted context length: {avg_effective_length}, Extract context num: {num_extract_context}")

        num_effective_prompt, num_extract_context = eva_rouge(doc_ids, outputs, contexts)
        print(f"Num of effective prompt: {num_effective_prompt}, Extract context num: {num_extract_context}")

        num_effective_prompt, num_extract_context = eva_bleu(doc_ids, outputs, contexts)
        print(f"Num of effective prompt: {num_effective_prompt}, Extract context num: {num_extract_context}")

        num_effective_prompt, num_extract_context, avg_max_sim, avg_mean_sim = eva_embedding_similarity(doc_ids, outputs, contexts, embed_model=vector_embed_model(cfg, device=device), device=device)
        print(f"Num of effective prompt: {num_effective_prompt}, Extract context num: {num_extract_context}, Average max embedding similarity: {avg_max_sim}, Average mean embedding similarity: {avg_mean_sim}")


if __name__ == "__main__":
    main()