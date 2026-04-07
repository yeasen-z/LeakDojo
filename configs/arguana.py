from configs.config_base import make_dataset_config

arguana = make_dataset_config(
    name="arguana",
    type="Social Media/Chat",
    intro="ArguAna is a dataset for argument-retrieval in the BEIR benchmark, containing English queries and counter-argumentative texts.",
    data_file="./data/arguana/corpus.jsonl",
    llm_model="gpt-4.1-mini",
    batch_size=256,
)
