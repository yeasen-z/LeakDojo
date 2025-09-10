from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any, Tuple
from torch_geometric.data import Data
import torch
import re
from langchain.schema import Document
from transformers import pipeline
from .rag_utils import get_data_chunks
from configs import GraphBaseConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


def generate_ere_message(text: str, region: str) -> str:
    """
    生成用于实体关系抽取的 JSON 指令型 prompt
    """
    if region == "medical":
        prompt = f"""You are a medical knowledge graph extraction assistant.\nPlease extract **Symptom**, **Disease**, and **Drug** entities from the following text, and establish relationships among them.\nReturn only JSON, strictly adhering to the format below:\n\n{{\n  \"nodes\": [\n    {{\"id\": \"symptom_1\", \"type\": \"Symptom\", \"name\": \"muscle cramp\"}},\n    {{\"id\": \"disease_1\", \"type\": \"Disease\", \"name\": \"heart attack\"}},\n    {{\"id\": \"drug_1\", \"type\": \"Drug\", \"name\": \"Panadol\"}}\n  ],\n  \"relationships\": [\n    {{\"source\": \"symptom_1\", \"target\": \"disease_1\", \"type\": \"possible_sign_of\"}},\n    {{\"source\": \"drug_1\", \"target\": \"symptom_1\", \"type\": \"relieves\"}}\n  ]\n}}\n\nRequirements:\n- If symptoms appear in the text (e.g., dizziness, nausea, muscle cramp), classify them as **Symptom**\n- If diseases appear (e.g., heart disease, gastritis, flu), classify them as **Disease**\n- If drugs appear (e.g., Panadol, Aspirin), classify them as **Drug**\n- Use relationship types only from the following set:\n  - \"possible_sign_of\" (Symptom → Disease)\n  - \"relieves\" (Drug → Symptom)\n  - \"treats\" (Drug → Disease)\n  - \"possible_related_to\" (for any potentially associated entities when relationship is uncertain or indirect)\n\nText: {text}"""
    elif region == "general":
        prompt = f"""You are a general-domain knowledge graph extraction assistant.\nExtract **entities** and **relationships** from the text below.\nReturn ONLY valid JSON in this exact format:\n\n{{\n  \"nodes\": [\n    {{\"id\": \"entity_1\", \"type\": \"Person\", \"name\": \"Alice\"}},\n    {{\"id\": \"entity_2\", \"type\": \"Organization\", \"name\": \"Google\"}},\n    {{\"id\": \"entity_3\", \"type\": \"Location\", \"name\": \"New York\"}}\n  ],\n  \"relationships\": [\n    {{\"source\": \"entity_1\", \"target\": \"entity_2\", \"type\": \"works_for\"}},\n    {{\"source\": \"entity_2\", \"target\": \"entity_3\", \"type\": \"located_in\"}},\n    {{\"source\": \"entity_1\", \"target\": \"entity_3\", \"type\": \"possible_related_to\"}}\n  ]\n}}\n\nEntity Types (choose best fitting):\n- Person (e.g., Elon Musk, Marie Curie)\n- Organization (e.g., Apple, United Nations)\n- Location (e.g., Tokyo, Mars, Atlantic Ocean)\n- Product (e.g., iPhone, ChatGPT, Tesla Model 3)\n- Event (e.g., World War II, Olympics 2024, Product Launch)\n- Concept (e.g., Inflation, AI Ethics, Climate Change) — use sparingly\n\nRelationship Types (choose from this list only):\n- works_for (Person → Organization)\n- located_in (Organization/Event → Location)\n- leads (Person → Organization/Event)\n- produces (Organization → Product)\n- causes (Event/Concept → Event/Concept)\n- competes_with (Organization/Product → Organization/Product)\n- related_to (any clear association)\n- possible_related_to (when association is uncertain or indirect)\n\nText: {text}"""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    return messages


def parse_triplets_json_strict(triplets_text: str):
    """
    直接匹配 {"head": "...", "relation": "...", "tail": "..."} 格式的对象
    返回列表
    """
    pattern = r'\{\s*"head"\s*:\s*"([^"]*)"\s*,\s*"relation"\s*:\s*"([^"]*)"\s*,\s*"tail"\s*:\s*"([^"]*)"\s*\}'
    matches = re.findall(pattern, triplets_text)
    
    triplets = []
    for head, relation, tail in matches:
        triplets.append({
            "head": head,
            "relation": relation,
            "tail": tail
        })
    return triplets

def graph_ere_extraction_llm(chunk_docs: List[Document], cfg: GraphBaseConfig, device: str = 'cpu') -> List[Dict[str, Any]]:
    """
    使用 LLM 直接抽取实体和关系，返回列表 [(head, relation, tail)]
    """
    # Load LLM model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.datastorage.ere_extract_llm)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.datastorage.ere_extract_llm,
        device_map="auto",
        torch_dtype="auto"
    )


    triplets_list = []
    for doc in chunk_docs:
        text = doc.page_content
        message = generate_ere_message(text.replace("input:", "").replace("output:", ""))
        in_prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
        )

        # 解码并打印回复
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        triplets = parse_triplets_json_strict(response)
        triplets_list.append({
            'document': doc,
            'triplets': triplets
        })
    
    return triplets_list