from references import zeng24_config, zeng24_question
from rag_components import get_data_chunks, graph_docs2chunk, extract_triplets_batch
from tools import load_saved_data, eva_pub_pri_hitnum, eva_pii_hitnum, eva_repeat_context, eva_rouge, eva_bleu, eva_embedding_similarity
import torch
import os
import argparse

def main():
    docs = get_data_chunks(zeng24_config.Zeng24ChatDoctor())
    print(f"Total chunks: {len(docs)}")
    graph_chunk = graph_docs2chunk(docs, split_by_sentence=True, lowercase=True)
    print(f"Total graph text chunks: {len(graph_chunk)}")
    ere_list = extract_triplets_batch(graph_chunk)
    print(f"Total extracted entity lists: {len(ere_list)}")


if __name__ == "__main__":
    main()
