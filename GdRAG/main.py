from references import zeng24_config, zeng24_question
from rag_components import get_retriever, get_prompts, get_retrieved_contexts, run_llm
import torch
import os



if __name__ == "__main__":
    '''
    1. 配置文件，每个方法单独实现
    2. 问题生成，每个方法单独实现
    3. 根据配置文件生成retriever
    4. 根据retriever和问题生成prompt，并对应保存 contexts，question，prompts
    5. prompts进行summarize
    6. 根据prompts生成answers，并保存
    7. 评估模块，单独实现
    '''

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    cfg = zeng24_config.Zeng24ChatDoctor()    
    
    # 保存文件的目录
    if not os.path.exists(cfg.expconfig.output_dir):
        os.makedirs(cfg.expconfig.output_dir,exist_ok=True)

    questions = zeng24_question.get_question(**zeng24_question.zeng24_chatdoctor_q)

    if 1:
        questions = questions[:10]
        
    retriver = get_retriever(cfg, retrival_database_batch_size=512, device=device, force_rebuild=False)

    prompts = get_prompts(cfg, retriver, questions, device=device)

    answers = run_llm(cfg, prompts, device)