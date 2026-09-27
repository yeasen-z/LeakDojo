# LeakDojo

**Evaluating RAG knowledge leakage and developing stronger extraction attacks.**

Official codebase for **LeakDojo: Decoding the Leakage Threats of RAG Systems**, published in **Findings of ACL 2026**.

[Paper](https://aclanthology.org/2026.findings-acl.287/) · [arXiv](https://arxiv.org/abs/2605.05818)

LeakDojo studies how attackers can induce retrieval-augmented generation (RAG) systems to disclose content from their knowledge bases. It supports controlled evaluation of RAG knowledge extraction attacks, with configurable attack strategies, language models, retrieval components, and defenses.

The paper evaluates **six existing attacks across fourteen LLMs and four datasets**. Building on this analysis, we introduce **logical masking instructions**, including **RankerSet** and **CodeClaim**, that disguise extraction intent as benign tasks.

## Logical Masking Attacks (LMA)

**Resistance to explicit extraction requests can overstate a RAG system's protection against leakage.**

### Attack Design

LeakDojo separates an attack into two components: a **query generator** that retrieves target content and an **adversarial instruction** that induces disclosure. LMA strengthens the instruction component by embedding extraction intent within a logical reasoning chain. The paper introduces **RankerSet** and **CodeClaim** as two instances.

This design allows the same query generator to be evaluated with default or logically masked instructions.

### Results with an Input Defense

With **DeepSeek-V3 on FIQA under the paper's T2 configuration**, Table 3 reports the following **Chunk Cumulative Leakage (CCL, %)** with the input intent detector enabled:

| Query generator | Default instruction | RankerSet | CodeClaim |
|-----------------|--------------------:|----------:|----------:|
| TGTB | 0.6 | 50.9 | 59.0 |
| GEN-PIDE | 0.2 | 47.8 | 59.6 |
| PoR | 0.2 | 51.7 | 57.9 |

CCL measures unique leaked chunks relative to the ideal maximum under the interaction budget; it is not per-query attack success rate. See Table 3 for results with output filtering as well.

**Takeaway:** Leakage evaluations should include logically masked instructions alongside explicit extraction requests.

### Attack Templates

[LMA templates](https://github.com/yeasen-z/LeakDojo/blob/main/attack_shop/adv_strings/leakdojo.json) are provided in `attack_shop/adv_strings/leakdojo.json`.

## Key Findings

- **Query generation and adversarial instructions make distinct contributions to leakage.** In our experiments, their contributions are approximately independent, and overall leakage is well approximated by their product. This supports evaluating these two parts of an attack separately.
- **Stronger instruction-following capability correlates with higher leakage risk.** Among the models evaluated, improved instruction following does not necessarily translate into better protection of retrieved content.
- **Improvements in RAG faithfulness can increase leakage risk.** Our results show that changes intended to improve faithful use of retrieved knowledge can also make protected content easier to extract. Answer quality and knowledge confidentiality therefore need separate evaluation.

These findings describe the settings evaluated in the paper; they are not universal claims about every model or RAG system. See the [paper](https://aclanthology.org/2026.findings-acl.287.pdf) for experimental conditions and full results.

## Evaluation Coverage

| Dimension | Coverage |
|-----------|----------|
| Paper evaluation | Six existing attacks, fourteen LLMs, and four datasets |
| Datasets | FIQA, SciFact, NFCorpus, and Enron Mail |
| Repository attack options | `wbtq`, `pide`, `tgtb`, `ikea`, `rtf`, `por`, and `dgea`; see Attack Methods |
| Configurable RAG components | Retrieval, query rewriting, reranking, and context extraction |
| Defense options | Intent filtering and output filtering |
| Leakage evaluation | ROUGE-L recall and the number of distinct corpus chunks extracted |

The six existing attacks are TGTB, GEN-PIDE, DGEA, RAG-Thief, PoR, and IKEA. The repository additionally provides the white-box target-query option (`wbtq`). LMA changes the adversarial instruction component; it is not an additional query-generator option.

## Using LeakDojo

Use the framework to:

- Compare knowledge extraction attacks under a shared RAG configuration.
- Examine how retrieval and generation components affect leakage.
- Evaluate leakage risks across models and document domains.
- Compare leakage with intent or output filtering enabled and disabled.

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


### 3. Prepare Data and Configure the Model Endpoint

Prepare a dataset following Datasets. Configure an OpenAI-compatible model endpoint and set `--llm_base_url`, `--llm_model`, and `--llm_api_key` to match your deployment. The command below retains the example endpoint from this repository.

### 4. Run Attacks

```bash
python main.py \
    --cfg_name fiqa --attack tgtb --attack_num 500 --batch_size 50 \
    --llm_model Qwen2.5-14B-Instruct \
    --llm_base_url http://localhost:22999/v1 --llm_api_key EMPTY \
    --reranker
```

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

### Datasets Used in the Paper

| Dataset | Type | Source | Config Name |
|---------|------|--------|-------------|
| **FIQA** | Finance | [BeIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa) | `fiqa` |
| **SciFact** | Academic/Research | [BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact) | `scifact` |
| **NFCorpus** | Medical | [BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus) | `nfcorpus` |
| **Enron Mail** | Email Corpus | [CMU Enron](https://www.cs.cmu.edu/~enron/) (May 7, 2015) | `enronmail` |

> **Enron Mail** requires manual download from CMU and preprocessing. See `tools/data_processor/enron_mail_to_corpus.ipynb`.

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

Adversarial instruction templates are stored under `attack_shop/adv_strings/`. LMA templates are available in `leakdojo.json`.

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

## Citation

If you use LeakDojo or build on its findings, please cite:

```bibtex
@inproceedings{zhang2026leakdojo,
  title = {LeakDojo: Decoding the Leakage Threats of RAG Systems},
  author = {Zhang, Maosen and Dong, Jianshuo and Lu, Boting and Li, Wenyue and Zhang, Xiaoping and Zhang, Tianwei and Qiu, Han},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year = {2026}
}
```
