#!/bin/bash
# LeakDojo - Attack Examples
# Usage: bash scripts/run_examples.sh <BASE_URL> <API_KEY> <MODEL_NAME>

BASE_URL="${1:?Usage: bash scripts/run_examples.sh <BASE_URL> <API_KEY> <MODEL_NAME>}"
API_KEY="${2:?Usage: bash scripts/run_examples.sh <BASE_URL> <API_KEY> <MODEL_NAME>}"
MODEL_NAME="${3:?Usage: bash scripts/run_examples.sh <BASE_URL> <API_KEY> <MODEL_NAME>}"

COMMON="--llm_base_url ${BASE_URL} --llm_api_key ${API_KEY} --reranker"

echo "===== pide + fiqa ====="
python main.py --cfg_name fiqa --attack pide --attack_num 2 --batch_size 1 ${COMMON} --llm_model ${MODEL_NAME} --lma_method reranker_role_play

echo "===== por + scifact ====="
python main.py --cfg_name scifact --attack por --attack_num 5 --batch_size 1 ${COMMON} --llm_model ${MODEL_NAME} --lma_method reranker_role_play

echo "===== tgtb + fiqa ====="
python main.py --cfg_name fiqa --attack tgtb --attack_num 2 --batch_size 5 ${COMMON} --llm_model ${MODEL_NAME}

echo "===== ikea + nfcorpus ====="
python main.py --cfg_name nfcorpus --attack ikea --attack_num 2 --batch_size 1 ${COMMON} --llm_model ${MODEL_NAME}

echo "===== rtf + scifact ====="
python main.py --cfg_name scifact --attack rtf --attack_num 2 --batch_size 1 ${COMMON} --llm_model ${MODEL_NAME}

echo "===== wbtq + fiqa ====="
python main.py --cfg_name fiqa --attack wbtq --attack_num 5 --batch_size 5 ${COMMON} --llm_model ${MODEL_NAME}

echo "===== dgea + fiqa ====="
python main.py --cfg_name fiqa --attack dgea --attack_num 5 --batch_size 1 ${COMMON} --llm_model ${MODEL_NAME}

echo "===== All done ====="
