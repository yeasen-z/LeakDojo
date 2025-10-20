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
pip install regex
pip install vllm==0.9.2
```

需要按以下指令安装 flash_attn
```bash
# pip install flash_attn==2.5.8 --no-cache-dir --use-pep517
pip install flash_attn==2.7.3 --no-cache-dir
```

然后安装其他的包内容:
```bash
pip install transformers==4.52.0
```

```bash
pip install langchain sentence-transformers rouge_score fire nltk pandas joblib chromadb modelscope chardet langchain_community FlagEmbedding langchain_huggingface rank_bm25
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
    
3. 运行的时候，尽量使用环境限制语句指定GPU id，用来限定embedding模型的加载位置，如下
    ```bash
        CUDA_VISIBLE_DEVICES=1 python main.py ...
    ```
4. 启动vllm模型服务，使用openai的接口\
    - 非think模型：
    ```bash
        CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
            --model ./Models/Qwen2.5-14B-Instruct \
            --port 22999 \
            --tensor-parallel-size 2 \
            --dtype auto \
            --gpu-memory-utilization 0.7
    ```
    - think模型
    ```bash
        CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
            --model ./Models/Qwen3-4B \
            --port 22999 \
            --tensor-parallel-size 2 \
            --dtype auto \
            --gpu-memory-utilization 0.7 \
            --enable-reasoning \
            --reasoning-parser deepseek_r1
    ```
    - 更多的可以在`scripts/`路径下参考脚本
5. 在slurm上启动vllm服务
    - 查看scripts中的脚本
    - 如果遇到查询不到slurm命令，那么运行"bash -l"
6. 在docker环境下，可能遇到ipc报错的问题，如下
    ```bash
        File "zmq/backend/cython/_zmq.py", line 1009, in zmq.backend.cython._zmq.Socket.bind
        File "zmq/backend/cython/_zmq.py", line 190, in zmq.backend.cython._zmq._check_rc
        zmq.error.ZMQError: Function not implemented (addr='ipc:///data/qiu_workspace/zms/tmp/03004154-d90f-4e92-9e06-4822d4115312')
    ```
    多半是 NFS 挂载盘 或者容器里的共享目录 → 不支持 ipc://

    可以尝试，明确其输出socket文件到本地 `export TMPDIR=/tmp`
7. 如果没有使用reranker，那么实际上rewriter发挥的作用可能很小，所以期望rewriter一定和reranker一起使用

## 代码结构

### 参考指令
1. 运行主流程（bbqg + rewriter + reranker + summarizer）
    ```
    python main.py \
    --device cuda:1 \
    --cfg_name fiqa \
    --llm_model ./Models/Qwen2.5-14B-Instruct \
    --llm_base_url http://localhost:22999/v1 \
    --llm_api_key EMPTY \
    --attack bbqg --attack_num 500 --batch_size 50 \
    --entity_file ./attack_shop/entity_base/FinTech.json \
    --rewriter --reranker
    ```
2. 运行主流程（wbtq，只做检索与回答）
    ```
    python main.py \
    --device cuda:1 \
    --cfg_name fiqa \
    --llm_model ./Models/Qwen2.5-14B-Instruct \
    --llm_base_url http://localhost:22999/v1 \
    --llm_api_key EMPTY \
    --attack wbtq --attack_num 200 --batch_size 50
    ```

### 说明

1. 实验设置保存在`configs/`文件路径下：
    - data: data_dir_list, description（影响改写/实体生成质量）
    - retrieval: method(top_k/fetch_k/score_threshold/top_n), embed.provider/model_dir
    - reranker.model 设置重排模型
2. 代码基于 vLLM server服务，服务需预先启动。详见 “注意事项”。主文件的LLM连接参数如下
    - --llm_model 模型名或本地路径
    - --llm_base_url OpenAI/vLLM 兼容地址
    - --llm_api_key API Key（本地 vLLM 常用占位值）
    - 生成控制：--llm_temperature, --llm_top_p, --llm_max_gen_len




### wbtq

采用[BEIR](https://github.com/beir-cellar/beir)的结构，可直接使用其公开数据集

### bbqg

### iter

