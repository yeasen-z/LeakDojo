# nohup python main.py --cfg_name fiqa --attack rtf --attack_num 200  --reranker --llm_model qwen3-235b-a22b   >  rtf_fiqa_010_200_qwen3235ba22b.log 2>&1 & 
# nohup python main.py --cfg_name nfcorpus --attack ikea --attack_num 200 --reranker --llm_model gemini-3-flash-preview   > ikea_nfcorpus_010_200_gemini-3-flash-preview.log 2>&1 & 
# nohup python main.py --cfg_name scifact --attack por --attack_num 200 --reranker --llm_model gpt-5.1   > por_scifact_010_200_gpt51.log 2>&1 & 
# nohup python main.py --cfg_name fiqa --attack pide --attack_num 200  --reranker --llm_model qwen3-8b   >  pide_fiqa_010_200_qwen38b.log 2>&1 & 
nohup python main.py --cfg_name scifact --attack ikea --attack_num 200 --reranker --llm_model gpt-5.1   > ikea_scifact_010_200_gpt51.log 2>&1 & 
