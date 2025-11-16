#!/bin/bash

# 设置：遇到错误不退出，继续执行
set +e

# BBQG attacks
nohup python main.py  \
    --llm_model ./Models/Qwen2.5-14B-Instruct \
    --llm_base_url http://localhost:22998/v1   \
    --cfg_name fever_filtered  \
    --rewriter --reranker \
    --attack bbqg --attack_num 500 --batch_size 25 \
    --entity_file ./attack_shop/entity_base/WikiHotGeneral.json  > bbqg_fever_filtered.log 2>&1 &

# WBTQ attacks
nohup python main.py \
    --llm_model ./Models/Qwen2.5-14B-Instruct \
    --llm_base_url http://localhost:22998/v1   \
    --cfg_name fever_filtered \
    --rewriter --reranker \
    --attack wbtq --attack_num 500 --batch_size 25 > wbtq_fever_filtered.log 2>&1 &

# IEGA attacks (using main_iter.py)
nohup python main_iega.py  \
    --llm_model ./Models/Qwen2.5-14B-Instruct \
    --llm_base_url http://localhost:22998/v1   \
    --cfg_name fever_filtered  \
    --rewriter --reranker \
    --attack iega --attack_num 500 --batch_size 25  > iega_fever_filtered.log 2>&1 &

# ICOA attacks
nohup python main_icoa.py  \
    --llm_model ./Models/Qwen2.5-14B-Instruct \
    --llm_base_url http://localhost:22998/v1   \
    --cfg_name fever_filtered  \
    --rewriter --reranker \
    --attack icoa --attack_num 500 --batch_size 25  > icoa_fever_filtered.log 2>&1 &