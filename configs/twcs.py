from configs.config_base import make_dataset_config

twcs = make_dataset_config(
    name="twcs",
    type="Social Media/Chat",
    intro="a social media corpus containing threads and related chat texts, useful for natural language processing tasks in the social media domain.",
    data_file="./data/twcs/threads.jsonl",
    llm_model="gpt-4.1-mini",
    batch_size=256,
)
