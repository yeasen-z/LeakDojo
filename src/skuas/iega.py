# Iterative Embedding-Guided Attack (IEGA)

import json
import random
import re
from collections import Counter
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
from langchain_huggingface import HuggingFaceEmbeddings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import textwrap

from src.interfaces import LLMManager, QueryGenerator
from src.utils import get_embed_model


class IKEAQueryGenerator(QueryGenerator):
    """A lightweight IKEA-style QueryGenerator.

    Responsibilities:
    - Manage an anchor-word pool via LLM
    - Turn anchor words into broad questions compatible with the pipeline
    - Provide optional utilities for similarity filtering
    """

    def __init__(
        self,
        llm: LLMManager,
        data_description: dict = None,
        embed_model_name: str = "./Models/all-mpnet-base-v2",
        device: str = "cuda:0",
    ):
        self.llm = llm
        self.embed_model = get_embed_model("hf", embed_model_name, device=device)
        self.data_description = data_description
        self.device = device

        # 候选以及数据存储
        self.full_query_db = [] # 外部候选池
        self.full_query_db_added_mask = np.array([], dtype=bool) # 标记该条目是否已被使用在query中
        self.queries = [] # 可以被使用的query队列
        self.query_valid_mask = np.array([], dtype=bool) # 标记该query是否使用过
        self.anchor_words_counts = dict() # 记录anchor words的使用频次

    def _generate_new_words(self, number, extra_demand: str = "", mode: str = "general"):
        # -- 生成新条目 -- #       
        new_texts = self.generate_anchor_word_with_llm(anchor_words_number=number, existed_words=self.queries, extra_demand=extra_demand, mode=mode)
        # 更新数据库添加entry
        self.add_entry_to_full_queryDB(new_texts)

    def add_entry_to_full_queryDB(self, texts: List[str]):
        """从外部列表初始化"""
        self.full_query_db.extend(texts)
        self.full_query_db_added_mask = np.concatenate([self.full_query_db_added_mask, np.zeros(len(texts), dtype=bool)]) 
        print(f"add entries (length:{len(texts)}) into full query DB...")

    def _add_query_entries(self, texts: List[str]):
        """添加新条目"""
        self.queries.extend(texts)
        # track entry usage status
        self.query_valid_mask = np.concatenate([self.query_valid_mask, np.zeros(len(texts), dtype=bool)]) 

    def shuffle_into_queries(self, prior_related_th:float=0.18, unsimilar_th:float=0.5):
        """筛选出未加入queryDB的与主题相关且相似度低于阈值的条目"""
        # shuffle unrelated to topics
        topic_sim_mtx = text_similarity_matrix(self.embed_model, self.full_query_db, [self.data_description["type"]])
        topic_valid_id = torch.nonzero(topic_sim_mtx.squeeze() > prior_related_th, as_tuple=True)[0].cpu().tolist()
        topic_valid_texts = [self.full_query_db[i] for i in topic_valid_id]
        # shuffle unsimilar
        unsimilar_texts, unsimilar_valid_idxs = find_unsimilar_texts(self.embed_model, topic_valid_texts, unsimilar_th, return_idx=True) 
        # find intersection
        valid_idxs = [topic_valid_id[i] for i in unsimilar_valid_idxs]
        # find unused
        not_added_idx = np.where(self.full_query_db_added_mask == False)[0].tolist() 
        unadded_valid_idxs = list(set(valid_idxs) & set(not_added_idx))
        to_add_texts = [self.full_query_db[idx] for idx in unadded_valid_idxs]
        # add entries
        self.full_query_db_added_mask[unadded_valid_idxs] = True
        self._add_query_entries(to_add_texts)
        # anchor word counter update
        new_anchor_word_dict = dict.fromkeys(to_add_texts, 1)
        self.anchor_words_counts = dict(Counter(new_anchor_word_dict) + Counter(self.anchor_words_counts))
        # verbose
        pre_topic=self.data_description["type"]
        print(f"筛选出{len(to_add_texts)}个与主题'{pre_topic}'相关且相似度低于{unsimilar_th}的条目, 当前可用query db长度{len(self.queries)}")
        
    def query(self, 
              score_k: int = 5,
              condition_match_mode: str = 'greedy',
              debug: bool = False,
              max_retries: int = 3,
              if_generate_new: bool = False,
              topic: str = None,
              generation_num: int = 100,
              extra_demand: str = None, 
              shuffle_topic_th: float = 0.05,
              shuffle_unsim_th: float = 0.4,
              sample_temperature: float=1) -> Optional[str]:
        """
        核心查询方法
        :param condition_match_mode: 在已满足条件的entry中的选择模式, 'random' or 'greedy' or 'soft_greedy
        :return: 找到的文本或None
        """
        
        if if_generate_new:
            self._generate_new_words(generation_num, extra_demand, mode='general')
            self.shuffle_into_queries(prior_related_th=shuffle_topic_th, unsimilar_th=shuffle_unsim_th)
        
        for _ in range(max_retries):
            valid_indices = np.where(~self.query_valid_mask)[0]
            if len(valid_indices) > 0:
                if condition_match_mode == "greedy":
                    prompts, scores_tensor, topk_indices = self.get_topk([self.queries[i] for i in valid_indices], k=1, return_indices=True, debug=debug)
                    best_idx = valid_indices[topk_indices[0]]
                    self.query_valid_mask[best_idx] = True
                    return prompts[0]
                
                elif condition_match_mode == "soft_greedy":
                    prompts, scores_tensor, topk_indices = self.get_topk([self.queries[i] for i in valid_indices], k=score_k, return_indices=True, debug=debug)
                    idx = random.choice([i for i in range(len(prompts))])
                    prompt = prompts[idx]
                    best_idx = valid_indices[topk_indices[idx]]
                    self.query_valid_mask[best_idx] = True
                    return prompt
                
                elif condition_match_mode == "softmax":
                    prompts, scores_tensor, topk_indices = self.get_topk([self.queries[i] for i in valid_indices], k=len(self.queries), return_indices=True, debug=debug)
                    probs = F.softmax(sample_temperature*scores_tensor, dim=0)
                    idx = torch.multinomial(probs, 1).item()
                    prompt = prompts[idx]
                    best_idx = valid_indices[topk_indices[idx]]
                    self.query_valid_mask[best_idx] = True
                    return prompt
                
                elif condition_match_mode == "random":
                    if debug:
                        prompts, scores_tensor, topk_indices = self.get_topk([self.queries[i] for i in valid_indices], k=score_k, return_indices=True, debug=debug)
                    best_idx = random.choice(valid_indices)
                    self.query_valid_mask[best_idx] = True
                    return self.queries[best_idx]
            else:
                self._generate_new_words(generation_num, extra_demand, mode='general')
                self.shuffle_into_queries(prior_related_th=shuffle_topic_th, unsimilar_th=shuffle_unsim_th)
            
            # raise ValueError("No valid words in counter db.")
            # return None

    def get_topk(self, 
           query_prompts: List[str], 
           k: int = 5,
           batch_size: int = 4,
           return_metadata: bool = False,
           return_indices:bool = False,
           debug:bool = False) -> Union[tuple, dict]:
        """获取外部prompt列表中平均得分最高的前k项
        
        Args:
            query_prompts: 需要评估的外部prompt列表
            k: 返回结果数量
            batch_size: 计算批次大小
            
        Returns:
            (前k个prompt列表, 对应分数列表)
        """
        # 输入验证
        if k <= 0:
            raise ValueError("k必须大于0")
        if not query_prompts:
            raise ValueError("输入prompt列表不能为空")
        actual_k = min(k, len(query_prompts))
        if debug:
            prompt_sims, answer_sims, scores = self.compute_scores(query_prompts, batch_size, debug)
        else:
            scores = self.compute_scores(query_prompts, batch_size)
        avg_scores = scores.mean(dim=1)  # shape: [n_query]
        if scores.size(dim=1) == 0:
            avg_scores = torch.nan_to_num(avg_scores, nan=0.0, posinf=0.0, neginf=0.0)
        topk_scores, topk_indices = torch.topk(avg_scores, actual_k)
        prompts = [query_prompts[i] for i in topk_indices.cpu().tolist()]
        
        
        # for debug
        if debug:
            info_dicts = []
            for i in range(len(self.prompts)):
                for j in range(len(query_prompts)):
                    info_dict = {
                                'iteration':self.properties[i]['iter'],
                                'mutation_id':self.properties[i]['mutation_id'],
                                'query_id': j,
                                'avg_score':avg_scores[j].item(), 
                                'query': query_prompts[j], 
                                'past_prompt':self.prompts[i], 
                                'retrieved':self.answers[i],
                                'prompt_sims': prompt_sims[j][i].item(),
                                'answer_sims': answer_sims[j][i].item(),
                                'score': scores[j][i].item(), 
                                'refusal': self.properties[i]['is_refusal_answer'],
                                'repeat_rate': self.properties[i]['repeat_rate'],
                                'is_related':self.properties[i]['is_related'],
                                }
                    info_dicts.append(info_dict)
            info_dicts=pd.DataFrame(info_dicts)
            info_dicts.sort_values(by='avg_score', ascending=False)
            info_dicts.to_csv("/home/guest/rag-framework/logs/extracted_db_query_scores.csv",)
            # in-prompts sims
            prior_topic = "medicine and symptom"
            topic_sim_mtx = text_similarity_matrix(self.embedding_model, self.prompts, [prior_topic]).squeeze()
            with torch.no_grad():
                self.entry_prompt_sim = chunked_matmul(self.prompt_embeddings, 
                                        self.prompt_embeddings.T, 
                                        step=4)
                self.entry_answer_sim = chunked_matmul(self.answer_embeddings, 
                                        self.answer_embeddings.T, 
                                        step=4)
                self.debug_qa_sim = chunked_matmul(self.prompt_embeddings, self.answer_embeddings.T, step=4)
            db_info_dicts = []
            for i in range(len(self.prompts)):
                for j in range(len(self.prompts)):
                    if i==j:
                        continue
                    info_dict = {'iteration':self.properties[i]['iter'],
                                 'mutation_id':self.properties[i]['mutation_id'],
                                 'causal':bool(self.properties[i]['iter']>self.properties[j]['iter']),
                                 'is_mutation': self.properties[i]['is_mutation'],
                                'compared_iteration':self.properties[j]['iter'],
                                'prompt':self.prompts[i], 
                                'compared_prompt':self.prompts[j], 
                                'retrieved':self.answers[i],
                                'compared_retrieved':self.answers[j],
                                'prompt_sims': self.entry_prompt_sim[i][j].item(),
                                'answer_sims': self.entry_answer_sim[i][j].item(),
                                'topic_sim': topic_sim_mtx[i].item(),
                                'p_a_sim_diff': self.entry_prompt_sim[i][j].item() - self.entry_answer_sim[i][j].item(),
                                'q_pa_sim': self.debug_qa_sim[i][j].item(),
                                'refusal': self.properties[i]['is_refusal_answer'],
                                'repeat_rate': self.properties[i]['repeat_rate'],
                                'co_repeat_rate': repeat_num(self.properties[i]['retrieve_id'], self.properties[j]['retrieve_id']) if self.properties[i]['iter'] > self.properties[j]['iter'] else 0,
                                'is_related':self.properties[i]['is_related'],
                                }
                    db_info_dicts.append(info_dict)
            db_info_dicts=pd.DataFrame(db_info_dicts)
            db_info_dicts.to_csv("/home/guest/rag-framework/logs/extracted_db_entry_info.csv",)

        scores = topk_scores
        if return_metadata:
            results = []
            for idx in topk_indices:
                results.append({
                    "prompt": query_prompts[idx],
                    "score": avg_scores[idx].item(),
                    "related_entries": [
                        {"text": self.prompts[i], "similarity": scores[idx][i].item()}
                        for i in torch.topk(scores[idx], min(3, len(self.prompts))).indices.tolist()
                    ]
                })
            return results
        else:
            if return_indices:
                return (prompts, scores, topk_indices)
            return (prompts, scores)


    def generate_anchor_word_with_llm(self, anchor_words_number:int=10, existed_words:List[str]=None, extra_demand:str=None, mode:str='general'):
        if mode == 'general':
            anchor_words = self.generate_general_anchor_words(number=anchor_words_number, existed_words=existed_words, extra_demand=extra_demand)
        elif mode == 'specific':
            anchor_words = self.generate_specific_anchor(number=anchor_words_number, existed_words=existed_words, extra_demand=extra_demand)

        return list(set(anchor_words))

    def generate_general_anchor_words(self, number: int = 100, existed_words: List[str] = None, extra_demand: str="") -> List[str]:
        """生成新的通用anchor words"""

        if not existed_words == None:
            existed_words_str = ', '.join(existed_words)
            difference_constraint = f"""The anchor words should be different from the following words: 
            {existed_words_str} 
            
            Besides, """
        else:
            difference_constraint = ''

        if not extra_demand:
            extra_demand = ''
        else:
            extra_demand = '5. ' + extra_demand

        prompt = textwrap.dedent(f"""
                    Generate a structured list of {number} anchor words related to the topic: {self.data_description["type"]}. {difference_constraint}
                    Your generated anchor words must:
                    Be distinct and contextually relevant. And could reasonably appear in this domain.
                    Be specific, diverse, and meaningful within the database region (e.g., organizations, places, events, systems, or terms).
                    Think briefly. Limit internal reasoning to several sentences before giving the final answer.

                    Output format requirements:
                    - Output only the entity names.
                    - Do not repeat any items or steps in your reasoning and output.
                    - Each entity must appear on a separate line.

                    Optional additional constraint:
                    {extra_demand}

                    Example output:
                    output1
                    output2
                    ...
                """)
        
        response, _ = self.llm.infer(prompt)
        entities = response.strip().splitlines()
        return entities

    def generate_specific_anchor(self, number: int = 5, existed_words: List[str]=None, extra_demand:str=None) -> List[str]:
        """围绕某个anchor word生成多个相关问题"""
        if not existed_words == None:
            existed_words_str = ', '.join(existed_words)
            difference_constraint = f"""The anchor words should be different from the following words: 
            {existed_words_str} 
            
            Besides, """
        else:
            difference_constraint = ''
            
        if not extra_demand:
            extra_demand = ''
        else:
            extra_demand = '2. ' + extra_demand
        
        prompt = textwrap.dedent(f"""
                    Generate a structured list of {number} **anchor words or phases or short sentences** related to the topic: **{self.data_description["type"]}**. {difference_constraint}Your Generated anchor words should be:
                    1. Each must appear on a separate line, so it can be easily parsed programmatically.
                    {extra_demand}

                    Output format requirements:
                    - Output only the entity names.
                    - Do not repeat any items or steps in your reasoning and output.
                    - Each entity must appear on a separate line.

                    Example output:
                    output1
                    output2
                    ...
                """)
        
        response, _ = self.llm.infer(prompt)
        fishes = response.strip().splitlines()
        return fishes

    def generate():
        pass



def chunked_matmul(A:Tensor, B:Tensor, step:int, show_progress:bool=False):
    """Matrix multiply, but A is divided into chunks to reduce memory usage.
    Args:
        A: The matrix on the left.
        B: The matrix on the right.
        step: The number of rows of a chunk.
    Return:
        The product of A and B.
    """
    n = len(A)
    results = []
    lbs = tqdm.tqdm(range(0, n, step)) if show_progress else range(0, n, step)
    for lb in lbs:
        ub = min(n, lb+step)
        results.append(torch.matmul(A[lb:ub], B))
    return torch.concatenate(results, dim=0)

def text_similarity(model:HuggingFaceEmbeddings, text_0:Union[str,List[str]], text_1:Union[str,List[str]]) -> Tensor:
    """Compute the similarity of texts one by one."""
    embedding_0 = model.embed_documents(text_0)
    embedding_1 = model.embed_documents(text_1)
    return torch.linalg.vecdot(embedding_0, embedding_1)

def text_similarity_matrix(model:HuggingFaceEmbeddings, text_0:List[str], text_1:List[str], batch_size:int=4) -> Tensor:
    """Compute the similarity of texts n by n.
    Args:
        text_0: list of input text.
        text_1: list of input text.
        batch_size: the batch size when doing dot product.
    Return:
        A matrix representing the similarities."""
    embedding_0 = model.embed_documents(text_0)
    embedding_1 = model.embed_documents(text_1)
    embedding_0 = torch.tensor(embedding_0, dtype=torch.float32)
    embedding_1 = torch.tensor(embedding_1, dtype=torch.float32)
    return chunked_matmul(embedding_0, embedding_1.transpose(0, 1), batch_size)

def self_similarity_matrix(model:HuggingFaceEmbeddings, texts:List[str], batch_size:int=4) -> Tensor:
    """Equivalent to text_similarity_matrix(model, text, text, batch_size).
    But no repeated computation."""
    embeddings = model.embed_documents(texts)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    return chunked_matmul(embeddings, embeddings.transpose(0, 1), batch_size)


def index_bools(lst:List, index:Union[Tensor,List[bool]]):
    """Similar with indexing a Tensor with Tensor[bool]
    But works on lists and returns an iterator."""
    for item, keep in zip(lst, index):
        if keep:
            yield item

def find_unsimilar_texts(model:HuggingFaceEmbeddings, texts:List[str], thresh:Optional[float]=None, n_preserve:Optional[int]=None, batch_size:int=4, return_idx:bool=False) -> Union[List[str]|List[int]]:
    """Find the subset of texts that are not similar to each other.
    Args:
        texts: the list of text to process.
        thresh: if specified, this function returns the subset of texts where similarity between each entry is less than `thresh`.
        n_preserve: if specified, this function returns the `n_preserve` most unsimilar texts from the input. You need to specify one and only one of `thresh` and `n_preserve`.
        batch_size: the batch size when doing dot product.
    Returns:
        a list of texts that are not similar to each other.
    """
    assert (thresh is None) ^ (n_preserve is None), "You need to specify one and only one of `thresh` and `n_preserve`."
    n = len(texts)
    # whether to keep each entry
    keep = torch.ones(n, dtype=torch.bool)
    similarities = self_similarity_matrix(model, texts, batch_size)
    if n_preserve is not None:
        n_removed = n - n_preserve
        similarities_sum = torch.sum(similarities, dim=0)
        for i in range(n_removed):
            # greedly remove the entry that is most similar to others
            entry_to_remove = int(torch.argmax(similarities_sum))
            keep[entry_to_remove] = False
            similarities_sum[entry_to_remove] = 0
            similarities_sum -= similarities[entry_to_remove]
    if thresh is not None:
        # for an undirected graph, remove fewest nodes, so that there is no edges left.
        adj_mat = (similarities-torch.eye(n, device=similarities.device)) >= thresh
        degrees = torch.sum(adj_mat, dim=0)
        while torch.any(degrees>0):
            # greedly remove the entry that is most similar to others, i.e. the node with largest degree
            entry_to_remove = int(torch.argmax(degrees))
            keep[entry_to_remove] = False
            degrees[entry_to_remove] = 0
            degrees -= adj_mat[entry_to_remove].to(dtype=degrees.dtype)
        if return_idx:
            return list(index_bools(texts, keep)), torch.nonzero(keep, as_tuple=True)[0].cpu().tolist()
    return list(index_bools(texts, keep))

def repeat_num(ls_1,ls_2):
    coset = set(ls_1 + ls_2)
    return len(ls_1)+len(ls_2)-len(coset)