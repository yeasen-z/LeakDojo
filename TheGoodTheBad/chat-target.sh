#!/bin/bash

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=27000 run_language_model.py --ckpt_dir llama-2-7b-chat --temperature 0.6 --top_p 0.9 --max_seq_len 4096 --max_gen_len 256 --path "chat-target/Q-R-T-" ;
