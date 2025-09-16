from references import zeng24_config, zeng24_question
from rag_components import vector_retriever, get_prompts, vector_retrieved_contexts, run_llm, vector_embed_model, get_queries, get_queries_id
from tools import load_saved_data, load_split, evaluate_retriever, get_beir_test_results

import torch
import os
import argparse

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

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

   # Run inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = zeng24_config.Zeng24fiqa()    

    # Run evaluation
    doc_ids, outputs, contexts, question = load_saved_data(cfg)
    question_id = get_queries_id(cfg)

    qid_doc_ids={}
    for i, qid in enumerate(question_id):
        qid_doc_ids[qid]={"q":question[i],"docs":[{"id":doc_ids[i][j], "text":contexts[i][j]} for j in range(len(contexts[i]))]}

    embed_model = vector_embed_model(cfg, device=device)
    queries, qrels = load_split(os.path.join(cfg.datastorage.raw_data_dir[0], 'queries.jsonl'), os.path.join(cfg.datastorage.raw_data_dir[0], 'qrels/test.tsv'))
    print(f"Total question num: {len(queries)}, qrels num: {len(qrels)}")
    test_results = get_beir_test_results(qid_doc_ids, qrels)
    print(f"Total test question num: {len(test_results)}")

    beir_retriever_results = {}
    for k, v in tqdm(test_results.items()):
        embed_vec = embed_model.embed_documents([v["docs"][i]['text'] for i in range(len(v["docs"]))])
        query_vec = embed_model.embed_documents([v["q"]])[0]
        query_vec = np.array(query_vec).reshape(1, -1)
        doc_vecs = np.array(embed_vec)
        similarities = cosine_similarity(query_vec, doc_vecs).flatten()
        beir_retriever_results[k] = {v["docs"][i]["id"]: similarities[i].item() for i in range(len(v["docs"]))}

    metrics = evaluate_retriever(qrels, beir_retriever_results, k=3)
    print("Retriever evaluation metrics: ", metrics)

if __name__ == "__main__":
    main()