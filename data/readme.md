```
data/
├── enronmail/
├── fever/
├── fiqa/
├── nfcorpus/
├── scifact/
└── twcs/
```
# Datasets from BEIR

, which could download and directly use in code.

- fever [fever-huggingface](https://huggingface.co/datasets/BeIR/fever)
- fiqa [fiqa-huggingface](https://huggingface.co/datasets/BeIR/fiqa)
- nfcorpus [nfcorpus-huggingface](https://huggingface.co/datasets/BeIR/nfcorpus)
- scifact [scifact-huggingface](https://huggingface.co/datasets/BeIR/scifact)

# TWCS (Twitter Customer Support)

A mix of multi region data, which should be downloaded at [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter/data).

preprocess the data with [twcs_to_corpus.ipynb](../tools/data_processor/twcs_to_corpus.ipynb)

which should be rougly concate to threads for each chunk.

# enron-mail
, which should be downloaded at [cmu](https://www.cs.cmu.edu/~enron/), version May 7, 2015 was chose.

preprocess the data with [enron_mail_to_corpus.ipynb](../tools/data_processor/enron_mail_to_corpus.ipynb).