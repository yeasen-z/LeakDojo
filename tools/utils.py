import re
import os
import json
import pandas as pd
from collections import defaultdict
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


def load_qrels(path_tsv):
    """
    读取 qrels (train/dev/test) 文件
    输出: {qid: {docid: relevance}}
    """
    df = pd.read_csv(path_tsv, sep="\t", names=["qid", "docid", "score"], header=0)
    qrels = defaultdict(dict)
    for row in df.itertuples(index=False):
        qrels[str(row.qid)][str(row.docid)] = int(row.score)
    return dict(qrels)


def load_queries(path_jsonl, qrels):
    """
    从 queries.jsonl 里筛选出对应 split 的 queries
    输入:
        path_jsonl: queries.jsonl 文件路径
        qrels: 来自 load_qrels 的字典
    输出:
        {qid: query_text}
    """
    queries = {}
    with open(path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["_id"])
            if qid in qrels:   # 只保留 qrels 中出现的 qid
                queries[qid] = obj["text"]
    return queries


def load_split(path_queries, path_qrels):
    """
    综合 loader，一次性返回 (queries, qrels)
    """
    qrels = load_qrels(path_qrels)
    queries = load_queries(path_queries, qrels)
    return queries, qrels

def get_beir_test_results(results, qrels):
    """
    只保留 results 中出现在 qrels 里的 query-id
    """
    return {qid: results[qid] for qid in results if qid in qrels}


public_ragfile_list=["wikitxt"]

pii_func_map = {
    "email": find_email_addresses,
    "phone": find_phone_numbers,
    "url": find_urls
}

pii_check_list=["email", "phone", "url"] # 需要检测的敏感信息类型




