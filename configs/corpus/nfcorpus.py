from configs.config_base import make_dataset_config

nfcorpus = make_dataset_config(
    name="nfcorpus",
    type="Medical/Clinical",
    intro="a medical corpus containing clinical notes and related medical texts, useful for natural language processing tasks in the healthcare domain.",
    data_file="./data/nfcorpus/corpus.jsonl",
    llm_model="gpt-4.1-mini",
    batch_size=256,
)
