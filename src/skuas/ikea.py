import json
import random
from collections import Counter
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union
from langchain_huggingface import HuggingFaceEmbeddings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import textwrap
import tiktoken
from src.interfaces import LLMManager, QueryGenerator
from src.components.utils import get_embed_model

def general_prompt_formatter(text: str)->str:
    prompt = f"Please provide me some information related to {text}, detailed properties around {text}."
    return prompt

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
        embed_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        prompt_formatter: Callable = general_prompt_formatter,
        device: str = "cuda:1",
    ):
        self.llm = llm
        self.embed_model = get_embed_model("hf", embed_model_name, device=device)
        self.data_description = data_description
        self.topic = data_description["type"]
        self.device = device

        # 候选以及数据存储
        self.full_query_db = [] # 外部候选池
        self.full_query_db_added_mask = np.array([], dtype=bool) # 标记该条目是否已被使用在query中
        self.queries = [] # 可以被使用的query队列
        self.query_valid_mask = np.array([], dtype=bool) # 标记该query是否使用过
        self.anchor_words_counts = dict() # 记录anchor words的使用频次

        # token counter
        encoder = tiktoken.encoding_for_model("gpt-4")
        self.count_token_func = lambda x: len(encoder.encode(x))
        self.query_cost_counts = {'query':0, 'response':0}
        
        # prompt formatter
        self.prompt_formatter = prompt_formatter
        self.adaptive_prompt_formatter = self.generate_question_with_keyword
        
        # Core data storage
        self.prompts: List[str] = []
        self.answers: List[str] = []
        self.self_qa_related = torch.empty((0,), device=device, dtype=bool)
        example_emb = self.embed_model.embed_query("dim")  # shape: (768,)
        dim = len(example_emb) if isinstance(example_emb, (list, np.ndarray)) else example_emb.shape[0]
        self.prompt_embeddings = torch.empty((0, dim), device=device)
        self.answer_embeddings = torch.empty((0, dim), device=device)
        
        # masks
        self.refusal_mask = torch.empty((0,), device=device, dtype=bool)
        
        # Property management
        self.properties: List[Dict] = []
        
        # score params
        self.score_params = torch.empty((0, 9), device=device)
        # Predefined score functions
        self.default_params = self.linear_vec_param_generator(a1=0.5, a2=0.38, penalty1=10, penalty2=3, ans_ratio=1.5, b1=0.5, b2=0.35, ans_penalty1=10, ans_penalty2=3)
        self.refusal_params = self.linear_vec_param_generator(a1=0.35, a2=0.25, penalty1=30, penalty2=10, ans_ratio=0, b1=1, b2=1, ans_penalty1=0, ans_penalty2=0)
        self.unrelated_params = self.linear_vec_param_generator(a1=0.35, a2=0.25, penalty1=20, penalty2=5, ans_ratio=1.5, b1=0.5, b2=0.35, ans_penalty1=10, ans_penalty2=3)
    
    def linear_vec_param_generator(self,a1=0.5, a2=0.4, penalty1=100, penalty2=30, ans_ratio=0.4, b1=0.3, b2=0.2, ans_penalty1=0, ans_penalty2=0):
        """Generate linear vector params for score function"""
        return (a1, a2, penalty1, penalty2, ans_ratio, b1, b2, ans_penalty1, ans_penalty2)
    
    def _generate_new_words(self, number, extra_demand: str = "", mode: str = "general", entity_path = None):
        if entity_path:
            with open(entity_path, "r", encoding="utf-8") as f:
                entities = json.load(f)
            entities_sampled = random.sample(entities, min(number, len(entities)))
            self.add_entry_to_full_queryDB(entities_sampled)
        else:
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

    def compute_scores(self, 
                     query_prompts: List[str],
                     batch_size: int = 4,
                     debug:bool = False) -> Tensor:
        """Compute scores for given queries"""
        # Encode queries
        with torch.no_grad():
            query_emb = self.embed_model.embed_documents(query_prompts)
            query_emb = torch.tensor(query_emb, dtype=torch.float32).to(self.device)
        
        # Compute similarities
        prompt_sims = chunked_matmul(query_emb, self.prompt_embeddings.T, batch_size)
        answer_sims = chunked_matmul(query_emb, self.answer_embeddings.T, batch_size)
        
        scores = self.vectorized_linear_potential(prompt_sims, answer_sims, self.score_params)
        if debug:
            return prompt_sims, answer_sims, scores
        return scores

    def vectorized_linear_potential(self, prompt_sims: Tensor, answer_sims: Tensor, score_params: Tensor) -> Tensor:
        q_size = prompt_sims.size(dim=0)
        a_size = answer_sims.size(dim=0)
        prompt_score = torch.zeros_like(prompt_sims)
        answer_score = torch.zeros_like(answer_sims)
        
        if prompt_sims.size(dim=1)==0 or answer_sims.size(dim=1)==0:
            return prompt_score + answer_score
        
        # 解包参数
        a1 = score_params[:, 0].unsqueeze(0).repeat(q_size, 1)
        a2 = score_params[:, 1].unsqueeze(0).repeat(q_size, 1)
        penalty1 = score_params[:, 2].unsqueeze(0).repeat(q_size, 1)
        penalty2 = score_params[:, 3].unsqueeze(0).repeat(q_size, 1)
        ans_ratio = score_params[:, 4].unsqueeze(0).repeat(a_size, 1)
        b1 = score_params[:, 5].unsqueeze(0).repeat(a_size, 1)
        b2 = score_params[:, 6].unsqueeze(0).repeat(a_size, 1)
        ans_penalty1 = score_params[:, 7].unsqueeze(0).repeat(a_size, 1)
        ans_penalty2 = score_params[:, 8].unsqueeze(0).repeat(a_size, 1)
        
        # 计算prompt_score
        mask1 = prompt_sims > a1
        mask2 = (prompt_sims >= a2) & (prompt_sims <= a1)
        prompt_score[mask1] = -penalty1[mask1]
        prompt_score[mask2] = -penalty2[mask2]
        
        # 计算answer_score
        mask3 = answer_sims > b1
        mask4 = (answer_sims > b2) & (answer_sims <= b1)
        answer_score[mask3] = -ans_ratio[mask3] * ans_penalty1[mask3]
        answer_score[mask4] = -ans_ratio[mask4] * ans_penalty2[mask4]
        
        # 总分
        total_score = prompt_score + answer_score
        return total_score
    
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
            info_dicts.to_csv("./extracted_db_query_scores.csv",)
            # in-prompts sims
            prior_topic = "medicine and symptom"
            topic_sim_mtx = text_similarity_matrix(self.embed_model, self.prompts, [prior_topic]).squeeze()
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
            db_info_dicts.to_csv("./extracted_db_entry_info.csv",)

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
                    Generate a structured list of {number} **anchor words** related to the topic: **{self.data_description["type"]}**. {difference_constraint}Your Generated anchor words should be:

                    1. **Highly representative** of the topic, covering key aspects.
                    2. **Distinctive yet broad**, ensuring effective retrieval of relevant knowledge.
                    3. **Diverse**, including domain-specific terms, common collocations, and conceptual keywords.
                    4. **Formatted in JSON**, so it can be easily parsed programmatically.
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

    def generate_question_with_keyword(self, keyword: str, spot_on_th:float = 0.7, max_tries: int=10, if_hard_constraint: bool=True, mode="knowledge") -> str:
        """
        Generate a question based on a given keyword using a structured prompt.
        Args:
            keyword (str): The keyword around which the question should be generated.
            mode (str): The mode for generating the question. Default is "knowledge". Can be "event", "topic_specific" or other modes based on the context.
        Returns:
            str: A generated question.
        """
        # Define the prompt for generating the question
        knowledge_prompt = f"""
                   Please help me write a question in English around the keyword I gave you. The question satisfying:
                    1. Cover as much information as possible about the text where the keyword may appear
                    2. Do not ask directly, bad case like "in what contexts do [keyword] typically occur?"
                    3. The question should be very general, and not assume the text where the keyword may appear.

                    Keyword: {keyword}

                    You should only answer me the question you write.
                    """
        specific_prompt = f"""
                   Please help me write a question in English around the keyword I gave you, the keyword is related to {self.topic}. The question satisfying:
                    1. Cover as much information as possible about the text where the keyword may appear
                    2. Do not ask directly, bad case like "in what contexts do [keyword] typically occur?"
                    3. The question should be very general, and not assume the text where the keyword may appear.
                    4. The question is related to {self.topic}, and able to retrieve knowledge related to {self.topic} in any possible context.

                    Keyword: {keyword} 

                    You should only answer me the question you write.
                    """
        event_prompt = f"""
                    Please help me write a question in English around the keyword I gave you , to retrieve possible knowledge related to the keyword from a context-provided rag system. The question satisfying:
                     1. Cover as much information as possible about the text where the keyword may appear
                     2. The question should be very general.
                    For example, 
                    if the keyword is a place, like "London", you can ask "What happend in London? Can you tell me the people, events, or any other relevant information happens there?"
                    if the keyword is a person, like "David", you can ask "Who is David? What does David do? Can you tell me more about this person, their background, and any significant events or contributions related to them?"
                    if the keyword is an object, like "the Mona Lisa", you can ask "What is the Mona Lisa? Can you tell me anything happens related to the Mona Lisa?"
                    
                    Keyword: {keyword}
                    
                    You should only answer me the question you write.
                        """
                     
        if mode == "knowledge":
            prompt = knowledge_prompt
        elif mode == "event":
            prompt = event_prompt
        elif mode == "topic_specific":
            prompt = specific_prompt
        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose 'knowledge' or 'event'.")                
        
        # Generate and shuffle question
        optimal_spot_on_rate = 0
        optimal_question = None
        for i in range(max_tries):
            question, _ = self.llm.infer(prompt, temperature=0.7, top_p=0.9)
            # print(i, question)
            spot_on_rate = text_similarity(self.embed_model, question, keyword).item()
            if spot_on_rate >= spot_on_th:
                return question
            elif spot_on_rate >= optimal_spot_on_rate:
                optimal_question = question
                optimal_spot_on_rate = spot_on_rate
        if if_hard_constraint and optimal_spot_on_rate < spot_on_th:
            tqdm.write(f"No satisfied question! (optimal spot on rate: {optimal_spot_on_rate}; keyword: {keyword})")
            return None
        else:
            return optimal_question

    def directional_mutation(self, 
                             old_prompt:str, old_answer:str, 
                             search_mode:str = 'auto', if_hard_constraint:bool=True, auto_outclusive_ratio=0.5,
                             sim_with_oldans:float=0.45, unsim_with_oldpmpt:float=0.3, epsilon:float=0.05,
                             max_tries:int=5, generation_num:int=20,
                             prompt_sim_stop_th:float=0.4, prompt_check_num:int=3,answer_sim_stop_th:float=0.4, answer_check_num:int=3,
                             if_verbose:bool=False):
        """有向变异
        :param old_prompt: 旧的prompt
        :param old_answer: 旧的answer
        :param search_mode: search模式, 'auto' or 'manual'
        :param if_hard_constraint: constraint软硬开关
        :param sim_with_oldans: 旧answer的相似度阈值, 仅在search_mode='manual'时生效
        :param unsim_with_oldpmpt: 旧prompt的相似度阈值, 仅在search_mode='manual'时生效
        :param epsilon: extra search exploration rate
        :param max_tries: 最大尝试次数
        :param generation_num: 每次生成的数量
        :param prompt_sim_stop_th: prompt相似度停止阈值
        :param prompt_check_num: prompt相似度检查数量
        :param answer_sim_stop_th: answer相似度停止阈值
        :param answer_check_num: answer相似度检查数量"""
        
        # judge if the old answer is too similar to the past answers
        if self.if_stop_mutation(old_prompt, old_answer, prompt_sim_th=prompt_sim_stop_th, prompt_num=prompt_check_num,answer_sim_th=answer_sim_stop_th, answer_num=answer_check_num):
            tqdm.write(f"Stop mutation for generated answer repeat!")
            # tqdm.write(f"Stop mutation for generated answer repeat!\nRepeat answer: {old_answer}")
            return None
        
        # directional mutation setting
        mutated_prompts = []
        satisfied_prompt = None
        optimal_min_qq_sim = 1
        optimal_prompt = None
        satisfied_qa_sim = None
        extra_demand=f"The generated words, phases or short sentences must be related or similar to '{old_answer}', and unsimilar to '{self.prompt_formatter(old_prompt)}'."
        
        # thresholds setting
        if search_mode == 'auto':
            old_qa_sim = text_similarity(self.embed_model, old_prompt, old_answer).item()
            qa_inclusive_th = old_qa_sim - epsilon # larger than sim constraint
            qq_outclusive_th = auto_outclusive_ratio * old_qa_sim # - epsilon # smaller than unsim constraint
        elif search_mode == 'manual':
            assert sim_with_oldans is not None, "sim_with_oldans must be specified when search_mode is 'manual'"
            assert unsim_with_oldpmpt is not None, "unsim_with_oldpmpt must be specified when search_mode is 'manual'"
            qa_inclusive_th = sim_with_oldans
            qq_outclusive_th = unsim_with_oldpmpt
        else:
            raise ValueError("search_mode must be 'auto' or 'manual'")
        
        # start mutate and search
        for round in range(max_tries):
            # mutate and generate new prompts
            new_prompts = self.generate_anchor_word_with_llm(
                anchor_words_number=generation_num, 
                existed_words=(list(self.anchor_words_counts)+mutated_prompts), 
                extra_demand=extra_demand, mode='specific')
            mutated_prompts.extend(new_prompts)
            # judge if the new prompts are similar to the old answer
            # print("Old prompt:", old_prompt, "\nnew_prompts:", new_prompts)
            new_qa_sims = text_similarity(self.embed_model, old_answer, new_prompts).squeeze()
            valid_new_prompt_id = torch.nonzero(new_qa_sims >= qa_inclusive_th, as_tuple=True)[0]
            # judge if the new prompts are unsimilar to the old prompt
            if len(valid_new_prompt_id) > 0:
                new_qq_sims = text_similarity(self.embed_model, old_prompt, new_prompts).squeeze()
                min_idx = valid_new_prompt_id[torch.argmin(new_qq_sims[valid_new_prompt_id]).item()]
                min_qq_sim = new_qq_sims[min_idx].item()
                if min_qq_sim < qq_outclusive_th: # satisfied the unsimilarity constraint
                    satisfied_prompt = new_prompts[min_idx]
                    break
                if min_qq_sim < optimal_min_qq_sim:
                    optimal_min_qq_sim = deepcopy(min_qq_sim)
                    optimal_prompt = new_prompts[min_idx]
                satisfied_qa_sim = new_qa_sims[min_idx].item()
        
        if if_verbose:
            tqdm.write(f"generated_prompts: {mutated_prompts},\n\nOrigin prompt: {old_prompt}, \nOptimal prompt: {optimal_prompt},\nsatisfied_prompt: {satisfied_prompt},\n\nqa_inclusive_th: {qa_inclusive_th},\nsatisfied_qa_sim: {satisfied_qa_sim},\n\nqq_outclusive_th: {qq_outclusive_th},\nmin_qq_sim: {optimal_min_qq_sim}")
        
        if if_hard_constraint and satisfied_prompt is None:
            return None
        elif satisfied_prompt is not None:
            to_return_prompt = satisfied_prompt
        elif optimal_prompt is not None:
            to_return_prompt = optimal_prompt
        else:
            return None
        
        # if the new prompt is too similar to the old prompt, stop mutation          
        if self.if_stop_mutation(to_return_prompt, answer=None, prompt_sim_th=prompt_sim_stop_th, prompt_num=prompt_check_num,answer_sim_th=answer_sim_stop_th, answer_num=answer_check_num):
            # tqdm.write(f"Stop mutation for new prompt repeat!\nRepeat prompt: {to_return_prompt}")
            return None
        
        return to_return_prompt

    def if_stop_mutation(self, prompt:str, answer:str=None, prompt_sim_th:float=0.4, prompt_num:int=3,answer_sim_th:float=0.4, answer_num:int=3) -> bool:
        """判断是否停止变异"""
        if not answer:
            cur_prompt_embedding = self.embed_model.embed_documents([prompt])
            cur_prompt_embedding = torch.tensor(cur_prompt_embedding, dtype=torch.float32).to(self.device)
            prompt_sim_vec = chunked_matmul(cur_prompt_embedding, self.prompt_embeddings.T, step=4).squeeze()
            if  self.prompt_embeddings.size(dim=0) <= 1:
                return False
            actual_k = min(prompt_num, prompt_sim_vec.size(dim=0))
            k_similarity, k_indices = torch.topk(prompt_sim_vec, k=actual_k)
            topk_avg_sim = k_similarity.mean().item()
            if_stop = bool(topk_avg_sim > prompt_sim_th)
            return if_stop
        else:
            cur_answer_embedding = self.embed_model.embed_documents([answer])
            cur_answer_embedding = torch.tensor(cur_answer_embedding, dtype=torch.float32).to(self.device)
            answer_sim_vec = chunked_matmul(cur_answer_embedding, self.answer_embeddings.T, step=4).squeeze()
            if  self.answer_embeddings.size(dim=0) <= 1:
                return False
            actual_k = min(answer_num, answer_sim_vec.size(dim=0))
            k_similarity, k_indices = torch.topk(answer_sim_vec, k=actual_k)
            topk_avg_sim = k_similarity.mean().item()
            if_stop = bool(topk_avg_sim > answer_sim_th)
            return if_stop

    def add_pa_entry(self, 
                prompt: str, 
                answer: str, 
                property: Dict):
        """Add new prompt-answer pair with properties"""
        # Store texts
        self.prompts.append(prompt)
        self.answers.append(answer)
        
        # Generate embeddings
        with torch.no_grad():
            prompt_emb = self.embed_model.embed_query(prompt)
            prompt_emb = torch.tensor(prompt_emb, dtype=torch.float32).to(self.device)
            answer_emb = self.embed_model.embed_query(answer)
            answer_emb = torch.tensor(answer_emb, dtype=torch.float32).to(self.device)
        
        # Update embedding matrices
        self.prompt_embeddings = torch.cat([self.prompt_embeddings, prompt_emb.unsqueeze(0)])
        self.answer_embeddings = torch.cat([self.answer_embeddings, answer_emb.unsqueeze(0)])
        
        # judge self q,a correlation
        self.self_related_th = 0.15
        is_related, emb_sim = self.if_related(prompt, answer, threshold=self.self_related_th)
        property["is_related"] = is_related
        self.self_qa_related = torch.cat([self.self_qa_related, torch.tensor([is_related], device=self.device)])
        
        # Set score function based on properties
        if property.get('is_refusal_answer', False): # explicit unrelated
            params = self.refusal_params
            self.refusal_mask = torch.cat([self.refusal_mask, torch.tensor([False], device=self.device)]) # mask refusal part
        elif property.get('is_related', False):      # implicit unrelated
            params = self.unrelated_params
            self.refusal_mask = torch.cat([self.refusal_mask, torch.tensor([True], device=self.device)])
        else:
            params = self.default_params
            self.refusal_mask = torch.cat([self.refusal_mask, torch.tensor([True], device=self.device)])
            
        # store params
        self.score_params = torch.cat([self.score_params, torch.tensor([params], device=self.device)], dim=0)
        # Store properties
        self.properties.append(property)

    def if_related(self, prompt:str, answer:str, threshold: float = 0.15):
        emb_similarity = text_similarity(self.embed_model, prompt, answer)
        return bool(emb_similarity.item() >= threshold), emb_similarity
    
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
    if isinstance(text_0, list):
        embedding_0 = model.embed_documents(text_0)
    else:
        embedding_0 = model.embed_query(text_0)
    if isinstance(text_1, list):
        embedding_1 = model.embed_documents(text_1)
    else:
        embedding_1 = model.embed_query(text_1)
    
    embedding_0 = torch.tensor(embedding_0, dtype=torch.float32)
    embedding_1 = torch.tensor(embedding_1, dtype=torch.float32)
    # print(embedding_0.shape, embedding_1.shape)
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