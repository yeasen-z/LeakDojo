from src.components import OpenAILLM, LLMQueryRewriter, VectorRetriever, RerankerManager, LLMHybridExtractor, SimplePromptConstructor
from src.skuas import PoRQueryGenerator, Q_inject, ShuffleQuestionInjection, KB
from src.pipeline import RAGPipeline
import os
import json
from tqdm import tqdm
import time
from .utils import setup, chunked
import sys

RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"

import random
random.seed(42)
def get_jsonl_line_count(path: str) -> int:
    if not os.path.exists(path):
        return 0

    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def AtkPoRPipeline(cfg, args):

    llm, llm_tool, intent_filter, output_filter, query_rewriter, retriever, reranker, extractor, constructor = setup(cfg, args)

    rag_pipeline = RAGPipeline(llm, 
                               query_rewriter, 
                               retriever, 
                               reranker, 
                               extractor, 
                               constructor, 
                               intent_filter, output_filter,
                               cfg, args)
    
    por_attacker = PoRQueryGenerator(llm_tool)

    output_dir = cfg.generate_exp_path(args.llm_model)
    os.makedirs(output_dir, exist_ok=True)
    jsonl_filename = cfg.generate_exp_filename(args, "por_attack")
    save_path = os.path.join(output_dir, jsonl_filename)

    max_extraction_iteration = args.attack_num
    nr_topics_to_use = 3    
    batch_size = args.batch_size
    kb_mirror=KB()

    count = get_jsonl_line_count(save_path)
    print(f"Resume from {count} records")

    query_count = count
    pbar = tqdm(total=max_extraction_iteration, desc="PoR Extraction",  initial = query_count)

    while len([data["relevance"][-1] for data in por_attacker.anchors_register.anchors_status.values() if data["relevance"][-1] > 0]) > 0:
        if query_count >= max_extraction_iteration:
            break
        # Dictionary to store all information for the current batch.
        batch = {
            "informations":[{"A_t":[], "generated_prompt": "", "llm_input":"", "chunks":[], "topics": [], "chunk_status":[], "added_topics": [], "command_repetition_chunks":[]} for _ in range(batch_size)],
            "times": {"gen_questions_time_s_avg_batch": 0, "gen_topics_time_s_avg_batch": 0, "target_attack_time_s_avg_batch": 0, "kb_mirror_operations_time_s_avg_batch":0, "topic_register_operations_time_s_avg_batch":0},
            "attack_status": {"stolen_chunks": 0, "stolen_topics": 0}
        }
        
        # 1. Relevance-based sampling of anchors.
        picks = por_attacker.anchors_register.get_A_t(nr_topics_to_use, batch_size)
        for index in range(batch_size):
            batch["informations"][index]["A_t"] = picks[index]
        
        # 2. Generate base queries from the sampled anchors.
        gen_questions_time_start = time.time()
        # print("batch[\"informations\"][index][\"A_t\"]  is",batch["informations"][index]["A_t"] )
        prompts = por_attacker.Q_generator([batch["informations"][index]["A_t"] for index in range(batch_size)], temperature=0.8)
        gen_questions_time_end = time.time()
        batch["times"]["gen_questions_time_s_avg_batch"] = (gen_questions_time_end - gen_questions_time_start)/batch_size
        
        for index in range(batch_size):
            batch["informations"][index]["generated_prompt"] = prompts[index]
            print(f"Generated prompt for index {index}: {prompts[index]}")
        
        # 3. Attack the target RAG system.
        non_completed_elements = [i for i in range(batch_size)]
        good_extractions = []
        
        for index in range(batch_size):
            batch["informations"][index]["command_repetition_chunks"] = [-1 for _ in range(len(por_attacker.commands))]

        target_attack_time_start = time.time()
        # Iterate through the injection commands until one succeeds in extracting chunks.
        for idx_command, command in enumerate(por_attacker.commands):
            prompts_injected = [str(Q_inject(batch["informations"][index]["generated_prompt"], command, ShuffleQuestionInjection.Type1)) for index in non_completed_elements]
            # print(prompts_injected)

            print(f"Processing {len(prompts_injected)} injected prompts.")

            (cleaned_batch_queries, contexts, doc_ids, rag_prompt, answers, reasons, rewritten_queries_list, extracted_contexts) = \
                rag_pipeline.run(prompts_injected)
            
            print(f"Finished {len(prompts_injected)} injected prompts.")

            # print(f"{GREEN}[+]{RESET} Completed parsing for command '{command}'. {len(non_completed_elements)} elements remaining.")
            for i in range(len(answers)):
                # --- 结果保存 ---            
                result_record = {
                    "id": str(query_count), # 迭代次数作为唯一标识符
                    "query_with_template": prompts_injected[i],
                    "cleaned_query": cleaned_batch_queries[i],
                    "rewritten_queries": rewritten_queries_list[i] if args.rewriter else [None],
                    "contexts": contexts[i],
                    "doc_ids": doc_ids[i],
                    "extract_contexts": extracted_contexts[i] if args.extractor else [],
                    "prompt": rag_prompt[i],
                    "answer": answers[i],
                    "reason": reasons[i] if args.reasoning else None
                }
                query_count+=1
                pbar.update(1)
                with open(save_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result_record, ensure_ascii=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())

            # 4. Parse the output to extract chunks.
            for relative_index, absolute_index in enumerate(non_completed_elements):
                    batch["informations"][absolute_index]["command_repetition_chunks"][idx_command] = 0
                    # chunks = answers[relative_index].split("\n")
                    chunks = answers[relative_index].split("\n\n")
                    batch["informations"][absolute_index]["command_repetition_chunks"][idx_command] = len(chunks)
                    batch["informations"][non_completed_elements[relative_index]]["llm_input"] = cleaned_batch_queries[relative_index]
                    if len(chunks) > 0:
                        batch["informations"][non_completed_elements[relative_index]]["chunks"] = chunks
                        good_extractions.append(absolute_index)
                        non_completed_elements[relative_index] = None
            non_completed_elements = [index for index in non_completed_elements if index is not None]
            if len(non_completed_elements) == 0:
                    break
            
        target_attack_time_end = time.time()
        batch["times"]["target_attack_time_s_avg_batch"] = (target_attack_time_end - target_attack_time_start)/batch_size
        
        # 5. Add non-duplicate chunks to the attacker's knowledge base.
        kb_insertion_time_start = time.time()
        for index in range(batch_size):
            for idx_chunk, chunk in enumerate(batch["informations"][index]["chunks"]):
                # `add_knowledge` returns True if the chunk is new.
                batch["informations"][index]["chunk_status"].append(kb_mirror.add_knowledge(chunk))
        kb_insertion_time_end = time.time()
        batch["times"]["kb_mirror_operations_time_s_avg_batch"] = (kb_insertion_time_end - kb_insertion_time_start)/batch_size
        
        # 6. Extract new anchors from the newly stolen chunks.
        gen_anchors_time_start = time.time()
        batch_anchors, gen_topics_output = por_attacker.T_generator([batch["informations"][index]["chunks"] for index in range(batch_size)],  temperature = 0.1)
        gen_anchors_time_end = time.time()
        batch["times"]["gen_anchors_time_s_avg_batch"] = (gen_anchors_time_end - gen_anchors_time_start)/batch_size
        batch["informations"][index]["generated_topics_text"] = gen_topics_output
        
        for index in range(batch_size):
            batch["informations"][index]["anchors"] = batch_anchors[index]
        
        # 7. Update anchor relevance scores.
        anchors_register_time_start = time.time()
        for element_index in range(batch_size):
            if len(batch["informations"][element_index]["chunks"]) < 1:
                    batch["informations"][element_index]["added_topics"] = []
                    continue
            # Identify which chunks were duplicates.
            duplicated_chunks = []
            for index, chunk in enumerate(batch["informations"][element_index]["chunks"]):
                if not batch["informations"][element_index]["chunk_status"][index]:
                    duplicated_chunks.append(chunk)
            
            por_attacker.anchors_register.update_relevance(batch["informations"][element_index]["chunks"], duplicated_chunks, batch["informations"][element_index]["A_t"], batch["informations"][element_index]["anchors"], batch["informations"][element_index]["chunk_status"])
        anchors_register_time_end = time.time()
        batch["times"]["topic_register_operations_time_s_avg_batch"] = (anchors_register_time_end - anchors_register_time_start)/batch_size
    
        # --- Logging and Saving ---
        num_stolen_docs = len(kb_mirror.kb)
        num_topics = len(por_attacker.anchors_register.anchors_status.keys())
        batch["attack_status"]["stolen_chunks"] = num_stolen_docs
        batch["attack_status"]["stolen_topics"] = num_topics
                
        # Log metrics to wandb.
        status = f"Stolen Documents: {num_stolen_docs}, Topics: {num_topics}"
        
        print(status.ljust(80), end='\r')
        sys.stdout.flush()

    pbar.close()

    print(f"\n[SUCCESS] All results saved to {save_path}")

    return save_path