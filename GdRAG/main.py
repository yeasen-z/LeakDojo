from references import zeng24_config
from rag import get_retriever, get_retrieved_contexts, run_llm
import torch



if __name__ == "__main__":

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    cfg = zeng24_config.Zeng24ChatDoctor()
    # cfg = Zeng24Wikitxt()

    questions = zeng24_config.get_question(**zeng24_config.zeng24_chatdoctor_q)

    print(questions)
    
    # retriver = get_retriever(cfg, retrival_database_batch_size=512, device=device, force_rebuild=False)

    # all_prompts = ["Please tell how to alleviation my diabetes.", "Please tell me how to treat my stomachache."]  # Define your prompts here
    # context = get_retrieved_contexts(cfg, all_prompts,retriver, device=device)


    # answers = run_llm(cfg, all_prompts, device)
    # print(answers)
