# Green Dinosaur RAG based on LangChain

- data: raw data of database
- configs: diff rags sys config file
- rag_components: core python files
- retrieval_stores: RAG base
- references: baselines
- tools: evaluation and show

# 环境搭建
```bash
pip install langchain transformers sentence-transformers rouge_score fire nltk pandas joblib
```

# 安装vllm

```bash
pip install "vllm==0.9.2"
pip install "flash_attn==2.5.8"
```

# 注意事项
1. 在使用qwen2.5-14B-instruct-1m进行vllm推理的时候，要在其config文件中，删除以下的设置，才能正常加载运行
    ```json
    "dual_chunk_attention_config": {
        "chunk_size": 262144,
        "local_size": 8192,
        "original_max_position_embeddings": 262144
    }
    ```
2. 运行的时候，尽量使用环境限制语句指定GPU id，如下
    ```bash
        CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --mode inference
    ```

    