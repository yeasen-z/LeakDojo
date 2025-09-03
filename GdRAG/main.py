from references import Zeng24ChatDoctor
from rag import get_retrieval_database, run_llm
import torch



if __name__ == "__main__":

    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'


    cfg = Zeng24ChatDoctor()
    # cfg = Zeng24Wikitxt()

    # docs = load_files2docs(cfg.datastorage.raw_data_dir)
    # print(type(docs[0]), docs[0].metadata)

    # split_docs = chunk_documents(docs, cfg)
    # print(f"Number of chunks: {len(split_docs)}")

    # embedding_model = get_embed_model(cfg, device=device)
    # print(embedding_model)

    chroma_database = get_retrieval_database(cfg, retrival_database_batch_size=512, device=device, force_rebuild=False)

    all_prompts = ["Please tell me who you are.", "How to search a paper for research?"]  # Define your prompts here
    answers = run_llm(cfg, all_prompts, device)
    print(answers)

