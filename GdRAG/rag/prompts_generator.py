from configs import BaseConfig

def generate_prompts(cfg: BaseConfig, user_query: str) -> list:
    """
    Generate prompts for the LLM based on user query and configuration.
    """
    prompts = []
    for suffix in cfg.prompt.suffix:
        prompts.append(f"{user_query}{cfg.prompt.adhesive}{suffix}")
    return prompts

