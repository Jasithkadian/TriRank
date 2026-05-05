# TriRank: Hybrid Retrieval Framework (BM25 + BGE + ColBERTv2)

A training-free 3-stage hybrid retrieval pipeline combining lexical, dense, and token-level reranking to achieve high-precision retrieval performance on MS MARCO and BEIR benchmarks.

---

## 🧠 Architecture

<p align="center">
  <img src="images/pipeline.png" width="800"/>
</p>

<p align="center">
  <em>Figure 1: Overview of the TriRank Pipeline</em>
</p>

---

## 📄 Research Paper

This repository contains the official implementation of our research paper:

**"TriRank: A Hybrid Retrieval Framework Combining BGE-large-en-v1.5 and ColBERTv2 for High-Precision Information Retrieval"**

**Authors:**
- Kurian Jose (Galgotias University)
- Jasith Kadian (Galgotias University)
- Shikhar Kulshreshtha (Galgotias University)
- Mohd Mohsin Ali (Galgotias University)
- Manish Raj (Galgotias University)
- Ankur Gogoi (Galgotias University)

---

## 🚀 Key Contributions

- 3-stage hybrid retrieval pipeline:
  - BM25 (lexical retrieval)
  - BGE-large-en-v1.5 (dense retrieval)
  - ColBERTv2 (token-level reranking)
- Training-free architecture (zero fine-tuning required)
- Reciprocal Rank Fusion (RRF) for merging ranked results
- GPU-optimized PyTorch chunked exact search over 8.8M embeddings
- Achieved:
  - nDCG@10 = 0.4638
  - MRR@10 = 0.3825

---

## 💡 What is TriRank?

Modern Retrieval-Augmented Generation (RAG) systems often struggle with the fundamental trade-off between exact keyword recall (sparse retrieval) and semantic meaning (dense retrieval). This gap leads to downstream LLM hallucinations and factual inaccuracies in enterprise search.

TriRank addresses this by providing a training-free 3-stage hybrid retrieval pipeline that synergizes lexical precision, semantic understanding, and fine-grained token-level relevance into a single scalable architecture.

It integrates:
- **BM25 (via bm25s):** Memory-optimized exact keyword matching.
- **BGE-large-en-v1.5:** Dense semantic retrieval utilizing a GPU-optimized PyTorch chunked exact search to bypass ANN approximation errors.
- **ColBERTv2:** Fine-grained token-level late-interaction reranking applied to a fused candidate pool (top 50).

---

## 🔄 Pipeline Flow

```
Query → BM25 + Dense Retrieval → RRF Fusion → ColBERTv2 → Final Results
```

---

## ⚙️ Pipeline Breakdown

### Stage 1: Parallel Retrieval
- BM25 for exact keyword matching  
- BGE-large-en-v1.5 for semantic retrieval  

### Stage 2: Dense Retrieval Optimization
- PyTorch chunked exact search  
- Eliminates ANN approximation errors  

### Stage 3: Fusion
- Reciprocal Rank Fusion (RRF)  

### Stage 4: Reranking
- ColBERTv2 with token-level MaxSim scoring  

---

## 📊 Results

### MS MARCO

| Method | nDCG@10 | MRR@10 | Recall@100 |
|--------|---------|--------|------------|
| BM25 Baseline | 0.2286 | 0.1796 | 0.6335 |
| BGE-large Dense Exact | 0.4376 | 0.3619 | 0.8968 |
| BM25 + BGE (RRF Fusion) | 0.3547 | 0.2833 | 0.8801 |
| TriRank (RRF + ColBERTv2) | **0.4638** | **0.3825** | **0.8801** |

### BEIR Zero-Shot Benchmark

TriRank generalizes well across domains without task-specific fine-tuning:

| Dataset | nDCG@10 | MRR@10 |
|---------|---------|--------|
| SciFact | 0.6638 | 0.5904 |
| ArguAna | 0.3408 | 0.2273 |
| NFCorpus | 1.9529* | 0.5481 |
| TREC-COVID | 9.1724* | 0.9066 |

*\*Scores unnormalized due to multi-graded relevance scales.*

---

## 🛠️ Installation

```bash
git clone https://github.com/Jasithkadian/TriRank.git
cd TriRank
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch transformers datasets faiss-cpu pyserini sentence-transformers numpy pandas tqdm
```

---

## ▶️ Usage

Run notebooks in this order:

1. notebooks/00_setup.ipynb  
2. notebooks/01_bm25_baseline.ipynb  
3. notebooks/02_bge_dense_baseline.ipynb  
4. notebooks/02_bge_dense_exact.ipynb  
5. notebooks/03_rrf_fusion.ipynb  
6. notebooks/04_colbert_reranking.ipynb  
7. notebooks/05_ablations.ipynb  
8. notebooks/06_beir_benchmarks.ipynb  
9. notebooks/07_analytics.ipynb  

---

## 📁 Project Structure

```
trirank/
│── notebooks/
│── scripts/
│── docs/
│── images/
│── README.md
│── requirements.txt
```

---

## 📦 Datasets

- MS MARCO  
- BEIR Benchmark:
  - SciFact  
  - NFCorpus  
  - ArguAna  
  - TREC-COVID  

Dataset setup instructions are inside:
notebooks/00_setup.ipynb

---

## 🧪 Reproducibility

To reproduce results:
- Run notebooks sequentially  
- Use MS MARCO / BEIR datasets  
- Ensure GPU for dense retrieval  

Expected performance:
- nDCG@10 ≈ 0.4638  
- MRR@10 ≈ 0.3825  

---

## 📌 Additional Resources

- docs/TriRank_Project_Guide.md  
- docs/workflow.md  
- docs/memory_prompt.md  

---

## 📬 Contact

Jasith Kadian  
- Email: jasithkadian@gmail.com  
- GitHub: https://github.com/Jasithkadian

Kurian Jose
- Email: kurianjoseoff@gmail.com 
- Github: https://github.com/KurianJose7586
