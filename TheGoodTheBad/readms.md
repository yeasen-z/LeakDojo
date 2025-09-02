# 环境
除了这些，原始的这个库，还需要llama进行安装


# 1. 创建文档索引

```bash
export CUDA_VISIBLE_DEVICES=9
python retrieval_database.py \
--dataset_name="chatdoctor" \
--encoder_model="bge-large-en-v1.5"
```

# 2. 生成prompt
```bash
export CUDA_VISIBLE_DEVICES=9 python generate_prompt.py
```

# 3. 进行攻击
```bash
CUDA_VISIBLE_DEVICES=9 torchrun --nproc_per_node=1 --master_port=27000 run_language_model.py --ckpt_dir llama-2-7b-chat --temperature 0.6 --top_p 0.9 --max_seq_len 4096 --max_gen_len 256 --path "chat-target/Q-R-T-"
```

# 4. 分析结果
```bash

```

