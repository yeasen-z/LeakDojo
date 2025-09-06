import os
import json

from configs import BaseConfig
from rag_components import get_llm_output_file


def load_result_data(cfg: BaseConfig):
    # if output not exist, return is question
    res_path = os.path.join(cfg.expconfig.output_dir, get_llm_output_file(cfg))

    if not os.path.exists(res_path):
        raise FileNotFoundError(f"The file does not exist: {res_path}")        

    with open(res_path, 'r', encoding='utf-8') as f:
        outputs = json.load(f)

    with open(os.path.join(cfg.expconfig.output_dir, 'context.json'), 'r', encoding='utf-8') as f:
        contexts = json.load(f)
    with open(os.path.join(cfg.expconfig.output_dir, 'sources.json'), 'r', encoding='utf-8') as f:
        sources = json.load(f)
    with open(os.path.join(cfg.expconfig.output_dir, 'question.json'), 'r', encoding='utf-8') as f:
        question = json.load(f)
    # with open(os.path.join(cfg.expconfig.output_dir, 'prompts.json'), 'r', encoding='utf-8') as f:
    #     prompts = json.load(f)

    # # 在做指标测试的时候，需要统一进行一次展开，方便匹配
    # if type(contexts[0]) is list:
    #     contexts = [item for sublist in contexts for item in sublist]

    # if type(sources[0]) is list:
    #     sources = [item for sublist in sources for item in sublist]

    # k = len(sources) // len(outputs)
    # assert len(question) == len(outputs)
    # assert len(question) == len(prompts)
    # assert len(sources) == len(contexts)
    # assert len(contexts) == len(prompts) * k

    return sources, outputs, contexts, question