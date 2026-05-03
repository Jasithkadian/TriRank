# TriRank: Hybrid Retrieval Framework (BM25 + BGE + ColBERTv2)

A training-free 3-stage hybrid retrieval pipeline combining lexical, dense, and token-level reranking to achieve state-of-the-art retrieval performance on MS MARCO and BEIR benchmarks.


## 📄 Research Paper

This repository contains the official implementation of our research paper:

**"TriRank: A Hybrid Retrieval Framework Combining BGE-large-en-v1.5 and ColBERTv2 for High-Precision Information Retrieval"**

Authors:
- Jasith Kadian
- Kurian Jose

  


##  Key Contributions

- 🔹 3-stage hybrid retrieval pipeline:
  - BM25 (lexical retrieval)
  - BGE-large-en-v1.5 (dense retrieval)
  - ColBERTv2 (token-level reranking)

- 🔹 Training-free architecture (zero fine-tuning required)

- 🔹 Reciprocal Rank Fusion (RRF) for merging results

- 🔹 GPU-optimized PyTorch chunked exact search over 8.8M embeddings

- 🔹 Achieved:
  - nDCG@10 = 0.4638
  - MRR@10 = 0.3825
