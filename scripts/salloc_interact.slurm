salloc -N 1 -n 1 -p a01 --gres=gpu:2 --nodelist=g78 --cpus-per-task=32 --time=01:00:00

source ~/.bashrc
conda activate rag_vllm
cd /home/fit/qiuhan4/WORK/zms/rag-llm/


python -m vllm.entrypoints.openai.api_server \
    --model ./Models/gemma-3-12b-it \
    --port 22999 \
    --tensor-parallel-size 2 \
    --dtype auto \
    --gpu-memory-utilization 0.7


python -m vllm.entrypoints.openai.api_server \
    --model ./Models/Qwen2.5-14B-Instruct \
    --port 22999 \
    --tensor-parallel-size 2 \
    --dtype auto \
    --gpu-memory-utilization 0.7