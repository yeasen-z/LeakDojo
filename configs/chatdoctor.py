from configs.config_base import make_dataset_config

chatdoctor = make_dataset_config(
    name="chatdoctor",
    type="Medical/Clinical",
    intro="a medical question answering benchmark derived from real-world sources such as medical forums and healthcare articles. it enables models to understand and respond to medical inquiries accurately.",
    data_file="./data/chatdoctor/corpus.jsonl",
    llm_model="gpt-4.1-mini",
    batch_size=512,
)
