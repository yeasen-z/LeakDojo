import math
from collections import defaultdict

def dcg(relevances):
    """计算 DCG"""
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))

def ndcg_at_k(true_rels, pred_scores, k=10):
    """
    true_rels: {doc_id: relevance}
    pred_scores: {doc_id: score}
    """
    # 排序后的预测文档
    ranked = sorted(pred_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    gains = [true_rels.get(doc_id, 0) for doc_id, _ in ranked]
    dcg_val = dcg(gains)

    # 理想排序
    ideal = sorted(true_rels.values(), reverse=True)[:k]
    idcg_val = dcg(ideal)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0

def precision_at_k(true_rels, pred_scores, k=10):
    ranked = sorted(pred_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    hits = sum(1 for doc_id, _ in ranked if true_rels.get(doc_id, 0) > 0)
    return hits / k

def recall_at_k(true_rels, pred_scores, k=10):
    ranked = sorted(pred_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    hits = sum(1 for doc_id, _ in ranked if true_rels.get(doc_id, 0) > 0)
    total_rel = sum(1 for v in true_rels.values() if v > 0)
    return hits / total_rel if total_rel > 0 else 0.0

def average_precision(true_rels, pred_scores, k=10):
    ranked = sorted(pred_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    score, hits = 0.0, 0
    for idx, (doc_id, _) in enumerate(ranked, start=1):
        if true_rels.get(doc_id, 0) > 0:
            hits += 1
            score += hits / idx
    total_rel = sum(1 for v in true_rels.values() if v > 0)
    return score / total_rel if total_rel > 0 else 0.0


def evaluate(qrels, results, k=10):
    ndcg, _map, recall, precision = [], [], [], []
    for qid in qrels:
        if qid not in results:
            continue
        true_rels = qrels[qid]
        pred_scores = results[qid]

        ndcg.append(ndcg_at_k(true_rels, pred_scores, k))
        _map.append(average_precision(true_rels, pred_scores, k))
        recall.append(recall_at_k(true_rels, pred_scores, k))
        precision.append(precision_at_k(true_rels, pred_scores, k))

    return {
        "nDCG@{}".format(k): sum(ndcg)/len(ndcg),
        "MAP@{}".format(k): sum(_map)/len(_map),
        "Recall@{}".format(k): sum(recall)/len(recall),
        "Precision@{}".format(k): sum(precision)/len(precision),
    }
