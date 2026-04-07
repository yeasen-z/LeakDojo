from configs.config_base import make_dataset_config

fiqa = make_dataset_config(
    name="fiqa",
    type="Finance",
    intro="a financial sentiment analysis benchmark derived from real-world sources such as StockTwits posts and financial news headlines.it enables models to understand market sentiment and investor opinions in financial contexts.",
    data_file="./data/fiqa/corpus.jsonl",
    llm_model="gpt-4.1-mini",
    batch_size=256,
)
