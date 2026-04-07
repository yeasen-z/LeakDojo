from configs.config_base import make_dataset_config

scifact = make_dataset_config(
    name="scifact",
    type="Academic/Research",
    intro="a dataset of expert-written scientific claims paired with evidence-containing abstracts, and annotated with labels and rationales.",
    data_file="./data/scifact/corpus.jsonl",
    llm_model="gpt-4.1-mini",
    batch_size=256,
)
