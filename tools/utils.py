import re
import os
import json
from typing import Union
from configs import VectorBaseConfig, GraphBaseConfig
from rag_components import get_llm_output_file


def load_saved_data(cfg: Union[VectorBaseConfig, GraphBaseConfig]):
    # if output not exist, return is question
    res_path = os.path.join(cfg.expconfig.output_dir, get_llm_output_file(cfg))

    if not os.path.exists(res_path):
        raise FileNotFoundError(f"The file does not exist: {res_path}")        

    with open(res_path, 'r', encoding='utf-8') as f:
        outputs = json.load(f)

    with open(os.path.join(cfg.expconfig.output_dir, 'context.json'), 'r', encoding='utf-8') as f:
        contexts = json.load(f)
    with open(os.path.join(cfg.expconfig.output_dir, 'doc_ids.json'), 'r', encoding='utf-8') as f:
        doc_ids = json.load(f)
    with open(os.path.join(cfg.expconfig.output_dir, 'question.json'), 'r', encoding='utf-8') as f:
        question = json.load(f)

    return doc_ids, outputs, contexts, question


def find_email_addresses(text):
    # Enhanced regular expression pattern for matching a wider range of email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    # Find all occurrences of the email pattern
    email_addresses = re.findall(email_pattern, text)
    return email_addresses


def find_phone_numbers(text):
    # Enhanced regular expression pattern for matching a wider range of phone numbers
    phone_pattern = r'(\+?\d{1,3}[ -]?)?(\(?\d{1,4}\)?[ -]?)?[\d -]{7,15}'
    # Find all occurrences of the phone number pattern
    phone_numbers = re.findall(phone_pattern, text)
    return phone_numbers


def find_urls(text):
    # Enhanced regular expression pattern for matching a broader range of URLs
    url_pattern = r'(https?://)?www\.[a-zA-Z0-9-]+(\.[a-zA-Z]+)+(/[a-zA-Z0-9-._~:/?#\[\]@!$&\'()*+,;=]*)?'
    # Find all occurrences of the URL pattern
    urls = re.findall(url_pattern, text)
    # Join the URL components
    urls = [''.join(url) for url in urls]
    return urls


public_ragfile_list=["wikitxt"]

pii_func_map = {
    "email": find_email_addresses,
    "phone": find_phone_numbers,
    "url": find_urls
}

pii_check_list=["email", "phone", "url"] # 需要检测的敏感信息类型
