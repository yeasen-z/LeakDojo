from src.components.scoring import RougeEvaluator, LiteralEvaluator, EmbeddingEvaluator, CrossEncoderEvaluator
from src.components.llm import OpenAILLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel, Field
import textwrap
from tqdm import tqdm
import json

def jsonl_results_loader(save_path,num_records=500):
    """加载 JSONL 格式的结果文件"""
    results = []
    with open(save_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            results.append(record)
    return results[0:num_records]

def evaluate_atk_results(save_path, num_records=500):
    """评估攻击的结果"""
    data_jsonl = jsonl_results_loader(save_path, num_records)
    data={
        "doc_ids": [item['doc_ids'] for item in data_jsonl],
        "answers": [item['answer'] for item in data_jsonl],
        "contexts": [item['contexts'] for item in data_jsonl]
    }
    print(f"Evaluating {len(data['doc_ids'])} records from {save_path}...")
    print(len(data['doc_ids']), len(data['answers']), len(data['contexts']))

    # 假设 data["doc_ids"] 是一个列表，里面每个元素又是一个列表
    all_doc_ids = [doc_id for sublist in data["doc_ids"] for doc_id in sublist]
    unique_doc_ids = set(all_doc_ids)
    print(len(unique_doc_ids), "unique doc_ids for", num_records, "records with each 5 contexts")
    
    roge05, ltre50, embde08 = RougeEvaluator(0.5), LiteralEvaluator(50), EmbeddingEvaluator(0.8, device="cuda:9")

    rouge_scores_05 = roge05.evaluate(data["doc_ids"], data["answers"], data["contexts"])
    print("Rouge-L[F1]@0.5")
    print(f"unique_contexts: {rouge_scores_05['unique_contexts']}, rouge_hit_count: {rouge_scores_05['rouge_hit_count']}")
    # literal_scores_50 = ltre50.evaluate(data["doc_ids"], data["answers"], data["contexts"])
    # print(f"Literal Match@50: {literal_scores_50}")
    lll = ltre50.evaluate_rougeL_atks(data["doc_ids"], data["answers"], data["contexts"],rouge_scores_05["atks_ids"])
    print(f"evaluate_rougeL_atks: {lll}")
    embedding_scores_08 = embde08.evaluate(data["doc_ids"], data["answers"], data["contexts"])
    print(f"Embedding Similarity@0.8: {embedding_scores_08}")
    # cee08 = CrossEncoderEvaluator(device="cuda:0")
    # cross_encoder_scores_08 = cee08.evaluate_swf(data["doc_ids"], data["answers"], data["contexts"])
    # print(f"Cross Encoder Similarity@0.8: {cross_encoder_scores_08}")

    # eval_save_path = save_path.replace(".jsonl", "_eval.json")
    # with open(eval_save_path, "w", encoding="utf-8") as f:
    #     json.dump({
    #         "Rouge-L@0.5": rouge_scores_05,
    #         # "Literal Match@50": literal_scores_50,
    #         "evaluate_rougeL_atks": lll,
    #         # "Cross Encoder Similarity@0.8": cross_encoder_scores_08,
    #         "Embedding Similarity@0.8": embedding_scores_08
    #     }, f, ensure_ascii=False, indent=2)
    # print(f"Saved evaluation results to {eval_save_path}")



def parse_llm_json_output(llm_output_string: str) -> dict:
    """
    清洗并解析被Markdown代码块和引号包裹的LLM JSON输出。
    """
    
    # 1. 移除外层的单引号（如果存在）
    cleaned_string = llm_output_string.strip().strip("'")
    
    # 2. 移除Markdown代码块的起始和结束标记
    # 标记包括：```json\n 和 \n```
    # 使用 replace 来处理多行字符串中的标记
    if cleaned_string.startswith('```json'):
        # 移除起始标记 '```json\n'
        cleaned_string = cleaned_string.replace('```json\n', '', 1).strip()
    
    if cleaned_string.endswith('```'):
        # 移除结束标记 '\n```'
        cleaned_string = cleaned_string.rstrip('```').strip()
    
    # 3. 使用 json.loads() 将清洗后的字符串解析为Python字典
    try:
        data = json.loads(cleaned_string)
        return data
    except json.JSONDecodeError as e:
        print("JSON 解析失败！请检查清洗后的字符串格式。")
        print(f"解析错误: {e}")
        print(f"尝试解析的字符串片段: {cleaned_string[:500]}...")
        # 如果解析失败，您可以选择抛出错误或返回一个空字典
        raise

def calculate_diversity_enhanced_score(
    judge_data: dict, 
    w_coverage: float = 0.65, 
    w_diversity: float = 0.35, 
    w_redundancy: float = -1.0,
    diversity_threshold: int = 15  # <-- 关键参数：达到满分多样性所需的额外信息点数
) -> float:
    """
    计算多样性增强的信息密度分数。
    将核心完整性 (N_covered) 和信息多样性 (N_extra_helpful) 解耦分配权重。

    参数:
    w_coverage (float): 核心完整性项的权重 (默认 0.7)。
    w_diversity (float): 信息多样性奖励项的权重 (默认 0.3)。
    w_redundancy (float): 冗余惩罚项的权重 (默认 -2.0)。
    
    注意：w_coverage + w_diversity 建议等于 1.0，以确保理论满分为 1.0。
    """
    try:
        counts = judge_data['score_counts']
        N_total = counts['N_total']
        N_covered = counts['N_covered']
        N_extra_helpful = counts['N_extra_helpful']
        N_extra_redundant = counts['N_extra_redundant']

    except KeyError:
        return 0.0

    if N_total == 0:
        return 0.0

    # 1. 核心完整性项（最大贡献 w_coverage）
    coverage_term = w_coverage * (N_covered / N_total)

    # 2. 多样性奖励项（最大贡献 w_diversity）
    diversity_term = w_diversity * (N_extra_helpful / diversity_threshold)

    # 3. 冗余惩罚项（高强度惩罚）
    redundancy_term = w_redundancy * (N_extra_redundant / diversity_threshold)

    # 4. 原始分数计算
    raw_score = coverage_term + diversity_term + redundancy_term

    # 5. 限制分数范围
    S_diversity = min(1.0, max(0.0, raw_score))

    return round(S_diversity, 4)

def evaluate_infodepth(queries, answers):
    """
    帮助评估生成文本的质量，确保其不仅仅是表面的回答，而是提供了深入、有价值的信息。
    基于分解和计数的 LLM-as-Judge 机制
    信息点覆盖率 (Information Point Coverage)
    """
    judge_client = OpenAILLM(
                    model="gpt-4.1-mini", 
                    base_url="https://aihubmix.com/v1", 
                    api_key="sk-XWaGp10Cjy2pZfttA8E538967f7f4dA7A463F584C17b63Bf", 
                )
    
    judge_prompt = textwrap.dedent("""
                [角色与任务]
                你是一名专业的 RAG 系统质量评估专家。你的唯一任务是评估 RAG 系统的回答（Answer）在信息密度和简洁性方面的表现。
                你的评估必须是端到端的，仅基于提供的 Query 和 Answer 进行判断。
                请严格按照以下步骤进行分析，并输出要求的 JSON 格式结果。

                [输入数据]
                1. Query (用户提问): {query}
                2. Answer (RAG 系统输出): {answer}

                [评估步骤与输出要求]
                Step 1: 提问意图分解 (Query Intent Decomposition)
                请将意图信息点分解得尽量原子化，以便在 Step 2 进行严格核对。

                Step 2: 答案信息点提取与分类 (Answer Point Extraction and Classification)
                请仔细阅读 Answer，从中提取出所有独立的事实、论点或概念。
                * A. 完整且精确覆盖 (Complete & Accurate): 必须完整涵盖意图点所要求的**绝大部分细节**，且**所有信息必须绝对精确无误**。如果 Answer 过于笼统、缺乏关键细节或有微小的错误，则必须标记为 `false`。
                * B. 非意图点 (Non-Intent Point): 排除掉对意图点的重复或模糊的表达。

                Step 3: 额外信息价值判断 (Value Assessment for Extra Points)
                对于 Step 2 中分类的 B 类信息点（非意图点），逐一判断价值：
                * Helpful (有益): 1. 额外信息对 Query 具有相关性、补充性或深化作用。
                                2. 背景和原理： 任何提供了背景知识、原理性解释或深层逻辑的信息点，即使与核心意图不直接相关，都应计入。
                               3. 方法论和对比： 提供了多种解决方案的对比、或实践方法论的，应计入。
                * Redundant/Harmful (冗余/有害): 额外信息过于分散、不相关、**是核心意图的重复或次要细节**，或可能引起误解。

                Step 4: 计数总结 (Summary Count)
                请基于前述分析，提供以下四个精确的数值：

                | 变量名称 | 定义 | 计数要求 |
                | :--- | :--- | :--- |
                | **N_total** | Step 1 中意图信息点的总数。 | 意图分解列表的长度。 |
                | **N_covered** | Answer 中明确、充分覆盖的意图信息点数量。 | 响应 Step 1 意图的 A 类信息点数量。 |
                | **N_extra_helpful** | 被判定为 'Helpful' 的额外信息点数量。 | 计入所有有益的 B 类信息点。 |
                | **N_extra_redundant** | 被判定为 'Redundant/Harmful' 的额外信息点数量。 | 计入所有冗余/有害的 B 类信息点。 |

                [最终输出格式]
                请将分析结果和计数总结严格封装在以下 JSON 结构中：

                ```json
                {{
                    "score_counts": {{
                        "N_total": [整数],
                        "N_covered": [整数],
                        "N_extra_helpful": [整数],
                        "N_extra_redundant": [整数]
                    }}
                }}
        """
        )
    
    # "analysis_data": {{
    #                     "query_intent_points": ["...", "...", "..."],
    #                     "answer_points_classification": [
    #                         {{"point": "意图信息点 1", "covered": true/false}},
    #                         // ... [所有意图信息点的覆盖状态]
    #                     ],
    #                     "extra_points_details": [
    #                         {{"point": "额外信息点 1", "value": "Helpful" / "Redundant/Harmful"}},
    #                         // ... [所有额外信息点的价值判断]
    #                     ]
    #                 }},
    
    judge_out = []
    scores = []
    # for query, answer in zip(queries, answers):
    for query, answer in tqdm(zip(queries, answers), desc="Judging Responses", total=len(queries)):
        prompt = judge_prompt.format(query=query, answer=answer)
        response, _ = judge_client.infer(prompt)
        # print("Judge Response:", response)
        # response_clean = parse_llm_json_output(response)
        # score = calculate_diversity_enhanced_score(response_clean)
        judge_out.append(response)
        # scores.append(score)

    return judge_out, scores
