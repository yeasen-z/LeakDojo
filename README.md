# LeakDojo

Official codebase for **"LeakDojo: Decoding the Leakage Threats of RAG Systems"**, Findings of ACL 2026.

LeakDojo is a comprehensive framework for evaluating the robustness of Retrieval-Augmented Generation (RAG) systems against knowledge extraction attacks. It implements 7 attack methods, configurable RAG pipelines, and multiple evaluation metrics to systematically assess information leakage risks. It further introduces novel logic-bypassing attack instructions designed to evade guardrails, enabling more effective knowledge extraction. See [`attack_shop/adv_strings/leakdojo.json`](attack_shop/adv_strings/leakdojo.json) for the full set of LMA (Logical Masking Attack) templates.

```
@article{zhang2026leakdojo,
  title={LeakDojo: Decoding the Leakage Threats of RAG Systems},
  author={Zhang, Maosen and Dong, Jianshuo and Lu, Boting and Li, Wenyue and Zhang, Xiaoping and Zhang, Tianwei and Qiu, Han},
  booktitle={ACL (Findings)},
  year={2026}
}
```

---

## Table of Contents

- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Attack Methods](#attack-methods)
- [Quick Start](#quick-start)
- [Evaluation](#evaluation)

---

## Datasets

LeakDojo uses the [BEIR](https://github.com/beir-cellar/beir) data format. Place your dataset under `data/` following this structure:

```
data/
├── {dataset_name}/
│   ├── corpus.jsonl      # Document corpus
│   ├── queries.jsonl      # Query set
│   └── qrels/
│       ├── dev.tsv
│       ├── test.tsv
│       └── train.tsv
```

### Paper Included Datasets

| Dataset | Type | Source | Config Name |
|---------|------|--------|-------------|
| **FIQA** | Finance | [BeIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa) | `fiqa` |
| **SciFact** | Academic/Research | [BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact) | `scifact` |
| **NFCorpus** | Medical | [BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus) | `nfcorpus` |
| **Enron Mail** | Email Corpus | [CMU Enron](https://www.cs.cmu.edu/~enron/) (May 7, 2015) | `enronmail` |

> **Enron Mail** requires manual download from CMU and preprocessing. See `tools/data_processor/enron_mail_to_corpus.ipynb`.

---

## Project Structure

```
LeakDojo/
├── main.py                          # Main entry point
├── evaluate.py                      # Evaluation entry point
├── configs/                         # Configuration files
│   ├── config_base.py               # VRConfig class (all default settings)
│   ├── __init__.py                  # Config registry
│   └── corpus/                      # Per-dataset configs
│       ├── fiqa.py
│       ├── scifact.py
│       ├── nfcorpus.py
│       └── enronmail.py
├── src/
│   ├── interfaces.py                # Abstract interfaces for all components
│   ├── components/                  # RAG pipeline components
│   │   ├── llm.py                   # LLM inference (OpenAI-compatible API)
│   │   ├── retrieval.py             # Vector retrieval (Chroma), BM25, reranker, extractor
│   │   ├── prompts.py               # Query rewriting & prompt construction
│   │   ├── defense.py               # Intent filter & output filter (defenses)
│   │   ├── scoring.py               # Evaluation metrics (ROUGE-L, embedding similarity)
│   │   └── utils.py                 # Utility functions
│   ├── pipeline/                    # Attack pipeline orchestration
│   │   ├── rag.py                   # Core RAG pipeline
│   │   ├── attack_static.py         # Pipeline for wbtq / pide / tgtb
│   │   ├── attack_ikea.py           # Pipeline for IKEA
│   │   ├── attack_rtf.py            # Pipeline for RTF (RAG-Thief)
│   │   ├── attack_por.py            # Pipeline for PoR
│   │   ├── attack_dgea.py           # Pipeline for DGEA
│   │   ├── evaluation.py            # Evaluation functions
│   │   └── utils.py                 # Pipeline utilities
│   └── skuas/                       # Attack query generators
│       ├── wbtq.py                  # White-Box Target Query
│       ├── gen_pide.py              # PI-DE query generator
│       ├── ikea.py                  # IKEA query generator
│       ├── rtf.py                   # RAG-Thief query generator
│       ├── por.py                   # PoR query generator
│       └── dgea.py                  # DGEA query generator
├── attack_shop/
│   ├── adv_strings/
│   │   ├── baselines.json           # Baseline adversarial prompt templates
│   │   └── leakdojo.json            # LMA (Logical Masking Attack) templates
│   └── baselines/
│       ├── tgtb/                    # Entity files for target-based attacks
│       └── dgea_embedding_spaces/   # DGEA embedding statistics
├── data/                            # Dataset files (BEIR format)
├── dataBase/                        # Vector database storage (ChromaDB)
├── models/                          # Local models
├── results/                         # Experiment results (JSONL)
├── tools/
│   ├── data_processor/              # Dataset preprocessing notebooks
│   │   ├── enron_mail_to_corpus.ipynb
│   │   └── wikitext_to_corpus.ipynb
│   └── readme.md
├── scripts/
│   ├── run_examples.sh              # Example run script
│   └── run_evaluate.sh              # Evaluation run script
└── requirements.txt                 # Dependencies
```

---

## Configuration

Settings are managed through two layers: **dataset configs** and **CLI arguments**.

### Dataset Configs (`configs/corpus/`)

Each dataset has a config file created via `make_dataset_config()` (see `configs/config_base.py`). These control:

| Section | Key Parameters | Description |
|---------|---------------|-------------|
| `data` | `data_dir_list`, `description`, `force_rebuild`, `datastorage_tool` | Dataset path, metadata, rebuild flag, storage backend |
| `tool_llm` | `model`, `base_url`, `api_key`, `temperature`, `top_p` | LLM settings for internal tools (rewriter, query generation) |
| `retrieval` | `method`, `top_k`, `fetch_k`, `score_threshold`, `top_n` | Retrieval strategy and parameters |
| `retrieval.embed` | `provider`, `model_name`, `model_dir` | Embedding model configuration |
| `reranker` | `provider`, `model` | Reranker model configuration |
| `extractor` | `provider`, `model` | Context extractor model configuration |

**Retrieval methods**: `mmr` (Maximal Marginal Relevance), `similarity_score_threshold`, `bm25`

**Default models**:
- Embedding: `BAAI/bge-large-en-v1.5`
- Reranker: `BAAI/bge-reranker-large`

### CLI Arguments (`main.py`)

```
# Basic
--cfg_name          Dataset config name (fiqa, scifact, nfcorpus, enronmail)
--device            GPU device (default: cuda:1)

# LLM
--llm_model         LLM model name or local path
--llm_base_url      OpenAI-compatible API endpoint
--llm_api_key       API key
--llm_temperature   Generation temperature (default: 0)
--llm_top_p         Top-p sampling (default: 1)
--llm_max_gen_len   Max generation length (default: 2048)

# Attack
--attack            Attack method: ikea | rtf | pide | wbtq | por | dgea | tgtb
--attack_num        Number of attack queries (default: 200)
--batch_size        Batch size (default: 1)
--entity_file       Path to entity file (for tgtb)

# Optional Pipeline Components
--rewriter          Enable query rewriting
--reranker          Enable reranking
--extractor         Enable context extraction
--intent_filter     Enable intent filtering (defense)
--output_filter     Enable output filtering (defense)
--reasoning         Save reasoning content (for thinking models)

# Utility
--build_only        Only build retrieval database, then exit
```

### Example: Run All Attacks

```bash
bash scripts/run_examples.sh <BASE_URL> <API_KEY> <MODEL_NAME>
```

### Example: Run a Single Attack

```bash
# tgtb attack with rewriter and reranker on FIQA
python main.py \
    --device cuda:1 \
    --cfg_name fiqa \
    --llm_model Qwen2.5-14B-Instruct \
    --llm_base_url http://localhost:22999/v1 \
    --llm_api_key EMPTY \
    --attack tgtb --attack_num 500 --batch_size 50 \
    --entity_file ./attack_shop/baselines/tgtb/Random_wikitext.json \
    --rewriter --reranker
```

---

## Attack Methods

| Method | Description | Pipeline |
|--------|-------------|----------|
| **wbtq** | White-Box Target Query — loads queries directly from corpus | `AtkStaticPipeline` |
| **pide** | GEN-PIDE — generates domain-specific queries using LLM + entities | `AtkStaticPipeline` |
| **tgtb** | TGTB — generates targeted queries for specific entity types | `AtkStaticPipeline` |
| **ikea** | IKEA — iterative anchor-based exploration with directional mutation | `AtkIKEAPipeline` |
| **rtf** | RAG-Thief — reflection-based attack for follow-up query generation | `AtkRTFPipeline` |
| **por** | PoR — anchor-based relevance sampling for chunk extraction | `AtkPoRPipeline` |
| **dgea** | DGEA — embedding optimization attack using adversarial suffix perturbation | `AtkDGEAPipeline` |

Adversarial prompt templates are defined in `attack_shop/adv_strings/collection.json`, with method-specific templates selected automatically at runtime.

---

## Quick Start

### 1. Install Dependencies

> **Note**: Install `vllm` and `flash_attn` first to avoid version conflicts.

```bash
# Core dependencies (version-sensitive)
pip install torch==2.7.0 torchvision==0.22.0
pip install regex
pip install vllm==0.9.2
pip install flash_attn==2.7.3 --no-cache-dir
pip install transformers==4.52.0

# Remaining packages
pip install langchain sentence-transformers rouge_score fire nltk pandas \
    joblib chromadb modelscope chardet langchain_community \
    FlagEmbedding langchain_huggingface rank_bm25 tiktoken
```

### 2. Prepare Models

Download embedding and reranker models to `models/BAAI/`:
- [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)


### 3. Run Attacks

```bash
python main.py \
    --cfg_name fiqa --attack tgtb --attack_num 500 --batch_size 50 \
    --llm_model Qwen2.5-14B-Instruct \
    --llm_base_url http://localhost:22999/v1 --llm_api_key EMPTY \
    --reranker
```

---

## Evaluation

### Evaluate Attack Results

```bash
python evaluate.py <result_file.jsonl> --num_records 200
```

Outputs ROUGE-L scores at multiple thresholds (0.3, 0.5, 0.7, 0.9), unique chunk count, and query count.

```bash
# Or use the shell script
bash scripts/run_evaluate.sh <result_file.jsonl>
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| ROUGE-L Recall | Overlap between response and retrieved contexts |
| Unique Chunks | Number of distinct corpus chunks extracted |

---
