#!/bin/bash

# 设置：遇到错误不退出，继续执行
set +e

# BBQG attacks
python main.py \
    --cfg_name fiqa --reranker \
    --attack bbqg --entity_file ./attack_shop/entity_base/FinTech.json \
    --attack_num 500 --batch_size 25 > bbqg_reranker_1118_q25_14b.log 2>&1

python main.py \
    --cfg_name fiqa --rewriter --reranker \
    --attack bbqg --entity_file ./attack_shop/entity_base/FinTech.json \
    --attack_num 500 --batch_size 25 > bbqg_rewriter_reranker_1119_q25_14b.log 2>&1

python main.py \
    --cfg_name fiqa --rewriter --reranker --summarizer \
    --attack bbqg --entity_file ./attack_shop/entity_base/FinTech.json \
    --attack_num 500 --batch_size 25 > bbqg_rewriter_reranker_sum_1120_q25_14b.log 2>&1

# WBTQ attacks
python main.py \
    --cfg_name fiqa --reranker \
    --attack wbtq \
    --attack_num 500 --batch_size 25 > wbtq_reranker_1121_q25_14b.log 2>&1 &

python main.py \
    --cfg_name fiqa --rewriter --reranker \
    --attack wbtq \
    --attack_num 500 --batch_size 25 > wbtq_rewriter_reranker_1122_q25_14b.log 2>&1 &

python main.py \
    --cfg_name fiqa --rewriter --reranker --summarizer \
    --attack wbtq \
    --attack_num 500 --batch_size 25 > wbtq_rewriter_reranker_sum_1123_q25_14b.log 2>&1 &

# IEGA attacks (using main_iter.py)
python main_iter.py \
    --cfg_name fiqa --reranker \
    --attack iega \
    --attack_num 500 --batch_size 25 > iega_reranker_1124_q25_14b.log 2>&1 &

python main_iter.py \
    --cfg_name fiqa --reranker --rewriter \
    --attack iega \
    --attack_num 500 --batch_size 25 > iega_rewriter_reranker_1125_q25_14b.log 2>&1 &

python main_iter.py \
    --cfg_name fiqa --reranker --rewriter --summarizer \
    --attack iega \
    --attack_num 500 --batch_size 25 > iega_rewriter_reranker_summarizer_1126_q25_14b.log 2>&1 &