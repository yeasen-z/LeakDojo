#!/bin/bash
#SBATCH -J rag_vllm
#SBATCH -N 1
#SBATCH -p a01
#SBATCH -o logs/stdout.%j
#SBATCH -e logs/stderr.%j
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --priority=10086

# 激活环境
source ~/.bashrc
conda activate rag_vllm
cd /home/fit/qiuhan4/WORK/zms/rag-llm/

# 启动 vLLM server
# --model 后面换成你的模型路径或 Hugging Face 名字
# --host=0.0.0.0 可以让其他节点访问，默认端口是 8000
python -m vllm.entrypoints.openai.api_server \
    --model ./Models/gemma-3-1b-it \
    --host 0.0.0.0 \
    --port 22999 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    --dtype auto \
