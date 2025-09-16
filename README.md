# Green Dinosaur RAG based on LangChain

## RAG data
使用BEIR的数据格式，即文件树为:

```raw
data_name/
├── corpus.jsonl
├── queries.jsonl
└── qrels/
    ├── dev.tsv
    ├── test.tsv
    └── train.tsv
```

## 环境搭建
一步步安装

### 安装vllm需要注意版本配对问题，目前测试了0.9.2可以使用的对应安装如下,最好不要直接安装requirements.txt

```bash
pip install torch==2.7.0 torchvision==0.22.0
pip install vllm==0.9.2
```

需要按以下指令安装 flash_attn
```bash
pip install flash_attn==2.5.8 --no-cache-dir --use-pep517
```

然后安装其他的包内容:
```bash
pip install transformers==4.52.0
```

```bash
pip install langchain  sentence-transformers rouge_score fire nltk pandas joblib chromadb modelscope chardet langchain_community FlagEmbedding langchain_huggingface 
```

## 注意事项
1. 在使用qwen2.5-14B-instruct-1m进行vllm推理的时候，要在其config文件中，删除以下的设置，才能正常加载运行
    ```json
    "dual_chunk_attention_config": {
        "chunk_size": 262144,
        "local_size": 8192,
        "original_max_position_embeddings": 262144
    }
    ```
2. vllm设置vllm_parallel_size参数的时候，一定要注意attn head数要能够被整除，否则会出错
    - 
3. 运行的时候，尽量使用环境限制语句指定GPU id，如下
    ```bash
        CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py --mode inference
    ```
4. 启动vllm模型服务，使用openai的接口
    ```bash
        CUDA_VISIBLE_DEVICES=5,6 python -m vllm.entrypoints.openai.api_server \
            --model ./Models/Qwen2.5-1.5B-Instruct \
            --port 8888
    ```

    