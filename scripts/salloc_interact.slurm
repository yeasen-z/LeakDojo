salloc -N 1 -n 1 -p a01 --gres=gpu:2 --nodelist=g56 --cpus-per-task=64 --time=01:00:00
salloc -N 1 -n 1 -p a01 --gres=gpu:2 --cpus-per-task=64 --time=01:00:00

source ~/.bashrc
cd /home/fit/qiuhan4/WORK/zms/rag-llm/
conda activate rag_vllm


python -m vllm.entrypoints.openai.api_server \
    --model ./Models/gemma-3-12b-it \
    --port 22999 \
    --tensor-parallel-size 2 \
    --dtype auto \
    --gpu-memory-utilization 0.7

nohup python -m vllm.entrypoints.openai.api_server \
    --model ./Models/Qwen2.5-14B-Instruct \
    --host 0.0.0.0 \
    --port 22999 \
    --tensor-parallel-size 2 \
    --dtype auto \
    --gpu-memory-utilization 0.7 > vllm.log 2>&1 &

nohup python -m vllm.entrypoints.openai.api_server \
    --model ./Models/Qwen2.5-32B-Instruct \
    --port 22999 \
    --tensor-parallel-size 2 \
    --dtype auto \
    --gpu-memory-utilization 0.88 > vllm.log 2>&1 &

nohup python -m vllm.entrypoints.openai.api_server \
    --model ./Models/Qwen2.5-14B-Instruct \
    --port 22999 \
    --dtype auto \
    --gpu-memory-utilization 0.88 > vllm.log 2>&1 &