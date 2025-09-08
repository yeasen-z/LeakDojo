"""
================================================================================
Graph-RAG 架构说明
================================================================================

此文件实现基于 Graph-RAG 的检索增强生成（RAG）流程，包括：
1. 文本预处理
2. 实体识别（NER）
3. 关系抽取（Relation Extraction）
4. 图构建（Graph Construction）
5. 图表示学习（Graph Neural Network, GNN）
6. 图增强检索（Graph-Enhanced Retrieval）
7. LLM 上下文生成与答案输出

--------------------------------------------------------------------------------
完整流程说明
--------------------------------------------------------------------------------

1. 文本预处理
   - 对原始文本 / 文档集进行分段、去噪、去重
   - 可选分句处理，确保实体和关系抽取的准确性

2. 实体识别 (NER)
   - 工具：SpaCy、HuggingFace Transformers NER 模型、LLM
   - 输出文本中的实体节点，用作图的节点

3. 关系抽取 (Relation Extraction)
   - 工具：OpenIE、关系分类模型、LLM Prompting
   - 输出实体之间的关系三元组 (subject, relation, object)
   - 三元组用作图的边

4. 图构建 (Graph Construction)
   - 节点：文本段落或实体
   - 边：实体关系或语义相似度
   - 节点特征：文本嵌入（HuggingFaceEmbeddings、BGE 等）
   - 工具：PyTorch Geometric (PyG)、DGL、NetworkX

5. 图表示学习 (Graph Representation Learning)
   - 使用 GNN（如 GCN、GAT、GraphSAGE）对节点/子图进行嵌入增强
   - 输出增强后的节点向量，用于后续检索

6. 图增强检索 (Graph-Enhanced Retrieval)
   - 结合向量相似度和图结构，检索 top-K 相关节点或子图
   - 可进行 k-hop 邻居采样、多跳路径选择
   - 目标：生成更丰富、更结构化的 LLM 输入上下文

7. LLM 上下文生成与答案输出
   - 使用 LangChain 或 LangGraph 管理检索结果和子图
   - 将图增强子图和文本上下文转换为 LLM prompt
   - 调用 LLM（OpenAI、LLaMA 等）生成最终答案

--------------------------------------------------------------------------------
工具与框架推荐
--------------------------------------------------------------------------------
- 文本处理与实体识别：SpaCy, HuggingFace Transformers, NLTK
- 关系抽取：OpenIE, 关系分类模型, LLM Prompting
- 图构建与 GNN：PyTorch Geometric (PyG), DGL, NetworkX
- 向量嵌入：HuggingFaceEmbeddings, BGE, Sentence-BERT
- RAG 管理与 LLM：LangChain, LangGraph, OpenAI, LLaMA

--------------------------------------------------------------------------------
说明
--------------------------------------------------------------------------------
- 流程可全自动化，但关系抽取质量会直接影响 Graph-RAG 效果
- 子图可视化和路径解释增强可解释性
- 每个步骤可独立替换和优化
================================================================================
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any, Tuple
from torch_geometric.data import Data
import torch
import re
from langchain.schema import Document
from transformers import pipeline
from .build_retriver_utils import get_data_chunks, NER_MODELS
from configs import BaseConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def graph_docs2chunk(
    docs: List[Document], 
    split_by_sentence: bool = True, 
    lowercase: bool = True,
    max_tokens: int = 512
) -> List[Document]:
    """
    对加载的文档进行预处理，包括分句、去噪、去重、智能合并。
    
    Args:
        docs (List[Document]): 原始文档对象列表
        split_by_sentence (bool): 是否先按句分段
        lowercase (bool): 是否小写化
        max_tokens (int): 合并后每个chunk的最大长度（按词数粗略估算）
    
    Returns:
        List[Document]: 处理后的文档对象列表
    """
    processed_docs = []
    seen = set()

    for doc in docs:
        text = doc.page_content.strip()
        text = re.sub(r'\s+', ' ', text)  # 去掉多余空格
        if lowercase:
            text = text.lower()

        # 按句分割
        if split_by_sentence:
            units = re.split(r'(?<=[。！？.!?])\s*', text)
        else:
            units = [text]

        # 合并成 chunk
        chunk = []
        token_count = 0
        for u in units:
            u = u.strip()
            if not u:
                continue
            # 估算词数（这里简单用空格分词，后续可换成 tokenizer）
            words = u.split()
            if token_count + len(words) > max_tokens and chunk:
                # 输出一个chunk
                merged = " ".join(chunk).strip()
                if merged and merged not in seen:
                    processed_docs.append(Document(page_content=merged, metadata=doc.metadata))
                    seen.add(merged)
                # reset
                chunk = []
                token_count = 0
            chunk.append(u)
            token_count += len(words)

        # 处理最后的chunk
        if chunk:
            merged = " ".join(chunk).strip()
            if merged and merged not in seen:
                processed_docs.append(Document(page_content=merged, metadata=doc.metadata))
                seen.add(merged)

    return processed_docs


# def graph_extract_entities(cfg: BaseConfig, texts: List[str]) -> List[Dict[str, Any]]:
#     """
#     自动识别文本中的实体节点
    
#     Args:
#         texts (List[str]): 文本列表
#         model (str): 选择的NER模型, 不同的database领域需要不同的模型

#     Returns:
#         List[Dict]: 每段文本的实体列表
#             示例: [{"text": "Barack Obama", "label": "PERSON", "start": 0, "end": 12}, ...]
#     """
#     # 加载预训练的英文NER模型
#     ner_pipeline = pipeline("ner", model=NER_MODELS[cfg.datastorage.data_region], aggregation_strategy="simple")
#     results = []
#     for text in texts:
#         raw_ents = ner_pipeline(text)  # 直接传字符串
#         entities = []
#         for ent in raw_ents:
#             entities.append({
#                 "text": ent["word"],           # 实体文本
#                 "label": ent["entity_group"],  # 实体类型 (PER, ORG, LOC, MISC)
#                 "start": ent["start"],         # 开始位置
#                 "end": ent["end"],             # 结束位置
#                 "score": ent["score"]          # 置信度
#             })
#         results.append(entities)

#     return results

# def graph_extract_relations(texts: List[str], entities: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """
    自动识别实体之间的关系，形成三元组
    
    Args:
        texts (List[str]): 文本列表
        entities (List[Dict[str, Any]): 实体列表

    Returns:
        List[Tuple]: 实体三元组列表 (subject, relation, object)

    这里使用的是通用关系提取模型，实际上效果可能不好，建议根据领域微调或使用更强的模型，所以不进行实现了
    """

    pass

# def extract_triplets_from_text(text: str, tokenizer, model) -> List[Tuple[str, str, str]]:
#     """
#     从单个文本 chunk 抽取三元组，并做关系归一化
#     """
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
#     outputs = model.generate(**inputs, max_new_tokens=256)
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     # 解析三元组
#     pattern = r"<triplet>(.*?)\|(.*?)\|(.*?)<triplet>"
#     matches = re.findall(pattern, decoded)
#     return [(s.strip(), r.strip(), o.strip()) for s, r, o in matches]

# def graph_extract_ere_triples(chunk_docs: List[Document]) -> List[Tuple[str, str, str]]:
#     '''
#     接收文本列表，直接输出实体关系三元组
#     使用REBEL模型
#     '''
#     # REBEL 模型初始化
#     MODEL_NAME = "Babelscape/rebel-large"
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cuda")  # 或 "cpu"

#     all_triplets = []
#     for doc in chunk_docs:
#         text = doc.page_content.strip()
#         if text:
#             triplets = extract_triplets_from_text(text, tokenizer, model)
#             all_triplets.extend(triplets)
#     # 可选：去重
#     seen = set()
#     unique_triplets = []
#     for t in all_triplets:
#         if t not in seen:
#             unique_triplets.append(t)
#             seen.add(t)
#     return unique_triplets

def extract_triplets_batch(chunk_docs: List[Document], batch_size: int = 1, max_length: int = 512, max_new_tokens: int = 256) -> List[Tuple[str, str, str]]:
    """
    批量抽取 Document 列表中的实体-关系-实体三元组，GPU加速
    """
    # REBEL 模型初始化
    MODEL_NAME = "Babelscape/rebel-large"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cuda")  # 或 "cpu"
    
    all_triplets = []

    model.eval()
    for i in range(0, len(chunk_docs), batch_size):
        batch_docs = chunk_docs[i:i+batch_size]
        texts = [doc.page_content.strip() for doc in batch_docs if doc.page_content.strip()]
        if not texts:
            continue

        for text in texts:
            # 超长文本切片
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            chunks = [tokenizer.decode(token_ids[j:j+max_length]) for j in range(0, len(token_ids), max_length)]
            
            for chunk in chunks:
                inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_length).to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

                matches = re.findall(r"<triplet>(.*?)\|(.*?)\|(.*?)<triplet>", decoded)
                for s, r, o in matches:
                    all_triplets.append((s.strip(), r.strip(), o.strip()))

                # 清理显存
                del inputs, outputs
                torch.cuda.empty_cache()

    # 去重
    seen = set()
    unique_triplets = []
    for t in all_triplets:
        if t not in seen:
            unique_triplets.append(t)
            seen.add(t)
    return unique_triplets

def build_graph(entities: List[Dict[str, Any]], relations: List[Tuple[str, str, str]], embeddings: List[torch.Tensor]) -> Data:
    """
    将实体和关系构建为 PyG 图对象
    
    Args:
        entities (List[Dict]): 实体节点列表
        relations (List[Tuple]): 实体关系三元组
        embeddings (List[torch.Tensor]): 每个实体的向量表示

    Returns:
        Data: PyG 图数据对象
    """
    pass


def apply_gnn(graph: Data) -> torch.Tensor:
    """
    对图进行 GNN 表示学习，输出增强后的节点向量
    
    Args:
        graph (Data): PyG 图对象

    Returns:
        torch.Tensor: 节点增强向量
    """
    pass


# def graph_retrieval(node_embeddings: torch.Tensor, query: str, graph: Data, top_k: int = 5) -> List[int]:
#     """
#     根据问题检索最相关节点/子图
    
#     Args:
#         node_embeddings (torch.Tensor): 图中节点向量
#         query (str): 用户问题
#         graph (Data): PyG 图对象
#         top_k (int): 返回节点数

#     Returns:
#         List[int]: 检索到的节点索引列表
#     """
#     pass


