I've reviewed all your files carefully. Let me give you a complete, ordered workflow from scratch.

---

## GemCol — Complete Implementation Workflow

### Before You Write Any Code

Your folder structure, shared utils, and package installs are already defined in the memory prompt. Start by setting those up exactly as specified, since every notebook depends on `utils/`.

---

### Phase 1 — Environment & Data Setup

**Step 1: Create folder structure**
Run the folder creation cell from the memory prompt. All paths (`/workspace/gemcol_evaluation/`, `/workspace/data/`, `/workspace/checkpoints/`, `/workspace/results/`, `/workspace/figures/`, `/workspace/notebooks/`) must exist before anything else.

**Step 2: Install packages**
```
pip install rank-bm25 tqdm ipywidgets pyserini qdrant-client sentence-transformers
```
Also install ColBERT separately since it needs its own setup from the `colbert-ir/colbertv2.0` repo.

**Step 3: Copy the shared utils files** (`metrics.py`, `config.py`, `data_loader.py`, `retrieval.py`) into `/workspace/gemcol_evaluation/utils/` exactly as defined. Do not modify them.

**Step 4: Download MS MARCO dev set**
You need three files — `queries.dev.small.tsv`, `qrels.dev.small.tsv`, and `collection.tsv` (8.8M passages, ~3GB). Download to `/workspace/data/msmarco/`. The collection may come as a `.tar.gz` — Cell 5 in your notebook handles extraction.

---

### Phase 2 — Notebook 01: BM25 Baseline (Person A)

This is your first runnable notebook. Follow cells 5–9 from the memory prompt exactly.

**What it produces:**
- `bm25_run.json` saved to `/workspace/results/` — this is the handoff file for RRF later
- BM25 nDCG@10, MRR@10, Recall@100 metrics
- Latency per query using `LatencyProfiler`

Expected nDCG@10: **0.28–0.30**. Your paper shows 0.187 — that's lower than expected, worth double-checking your Pyserini setup vs rank-bm25 tokenization.

---

### Phase 3 — Notebook 02: BGE Dense Baseline (Person B)

**Setup Qdrant** (run locally via Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**What to do:**
1. Load corpus using `load_msmarco_dev()`
2. Encode all 8.8M passages with `BAAI/bge-large-en-v1.5` in batches of 256 — this will take several hours even on your 70GB GPU. Use `load_or_compute()` to checkpoint.
3. Index embeddings in Qdrant with HNSW parameters: `M=32`, `efConstruction=200`
4. Apply L2 normalization before indexing so cosine similarity = inner product
5. Run retrieval on all 6,980 dev queries, save `bge_run.json`

Expected nDCG@10: **0.35–0.40**

---

### Phase 4 — Notebook 03: RRF Fusion (Person A, needs bge_run.json)

Once both `bm25_run.json` and `bge_run.json` exist:

```python
from utils import rrf_fusion, load_run_json
bm25_run = load_run_json("/workspace/results/bm25_run.json")
bge_run = load_run_json("/workspace/results/bge_run.json")
fused_run = rrf_fusion([bm25_run, bge_run], k=60, top_k=100)
```

Save the fused run. Expected nDCG@10: **0.32–0.35**. This becomes the input to ColBERT.

---

### Phase 5 — Notebook 04: ColBERTv2 Reranking (Person B)

Take the fused ~100 candidates per query and rerank using `colbert-ir/colbertv2.0`.

The MaxSim scoring formula from your paper:
```
Score(Q, P) = Σᵢ maxⱼ cos(E(qᵢ), E(pⱼ))
```

Encode query tokens and passage tokens separately. For each query token, find the max cosine similarity against all passage tokens, then sum. Only apply this to the ~100 candidates — never the full corpus.

Call `torch.cuda.empty_cache()` after every batch.

Expected nDCG@10: **0.42–0.45** (your paper shows 0.371 — review Section IV-D in the draft, the reranking stage is currently just the formula with no explanation, which is likely a gap)

---

### Phase 6 — All Required Tests (from your Evaluations PDF)

Run these in this order, logging everything with `ExperimentTracker`:

#### Section A — Main Benchmarks (all → Table I and II)
| Test | Notebook | Output |
|---|---|---|
| MS MARCO nDCG@10, MRR@10, Recall@100 | Done in phases 2–5 | Table I |
| BEIR — NFCorpus | `07_beir_benchmarks.ipynb` | Table II |
| BEIR — SciFact | same notebook, run separately | Table II |
| BEIR — TREC-COVID | same | Table II |
| BEIR — Arguana | same | Table II |

**Critical BEIR rule:** Run one dataset at a time. Delete corpus after recording metrics. Order: NFCorpus → SciFact → TREC-COVID → Arguana. Use `beir_corpus_to_texts()` before passing to BM25 or BGE.

#### Section B — Baseline Comparisons (→ Table I + II)
You need four systems evaluated side by side:

1. **BM25 alone** — already done in Phase 2
2. **BGE dense alone** — already done in Phase 3
3. **BM25 + BGE + RRF (no reranking)** — done in Phase 4, just stop before ColBERT
4. **ColBERTv2 over BM25 only** — take BM25 top-100, rerank with ColBERT, no BGE. This proves first-stage quality determines reranking ceiling.

#### Section C — Ablation Studies (`05_ablations.ipynb` → Table III + Figure 3)

**Ablation 1 — Weaker embedding swap (MiniLM)**
Replace BGE-large with `MiniLM-L6-v2` (384-dim), keep everything else identical. Run full pipeline. This proves high-quality embeddings are the critical ingredient.

**Ablation 2 — Candidate pool size variation**
Run RRF → ColBERT with pool sizes k = 25, 50, 100, 200. Record nDCG@10 and latency for each. This produces Figure 3 (pool size vs latency tradeoff) and justifies your k=100 choice.

**Ablation 3 — RRF k parameter sensitivity**
Run RRF with k = 10, 30, 60, 100 (the smoothing constant, not pool size). Record nDCG@10 for each. This justifies your k=60 choice with actual data — currently your paper just cites Cormack et al. without empirical backing.

**Ablation 4 — Parallel vs sequential retrieval latency**
Run BM25 and BGE sequentially vs in parallel using threading. Record average latency. Quality should be identical — only latency differs.

#### Section D — Latency Profiling (→ Table I latency column)
Use `LatencyProfiler` as a context manager on each stage separately. Average over 100+ queries. Report mean, p95, p99. Targets from memory prompt: BM25 50–100ms, BGE 200–500ms, RRF <10ms, ColBERT 500–800ms, total <1500ms.

#### Section E — Original Analysis (differentiates your paper)

**Retrieval source breakdown → Figure 4**
Use `retrieval_source_breakdown()` from `retrieval.py`. For your final top-10 results across the dev set, count: what % of relevant passages came from BM25 only, BGE only, or both retrievers. This becomes a pie/bar chart proving you need both retrievers.

**Failure case analysis → Section VI of your paper**
Manually examine 15–20 queries where GemCol ranks the relevant passage outside top-10. Categorize into: ambiguous query, rare entity, multi-hop reasoning needed, query too short. Write one paragraph per category in your Discussion section.

**Query length vs performance (optional figure)**
Split dev queries into short (1–3 words), medium (4–7), long (8+). Report nDCG@10 for GemCol and BM25 in each group. Short queries favor BM25; long queries favor dense retrieval.

---

### Phase 7 — Figures and Tables (`08_figures_and_tables.ipynb`)

Run last, after all experiments are logged in `experiments.json`.

| Deliverable | Data source |
|---|---|
| Table I — MS MARCO metrics + latency | Phases 2–5 + latency profiling |
| Table II — BEIR results | Phase 6 Section A |
| Table III — Ablations | Phase 6 Section C |
| Figure 3 — Pool size vs latency | Ablation 2 |
| Figure 4 — Retrieval source breakdown | `retrieval_source_breakdown()` |

---

### Paper Writing — What's Missing Right Now

Looking at your draft (`draft_incomplete_paper.pdf`), Section IV exists but is very thin. Based on the proposed method notes you uploaded, here's what needs to be written or expanded:

- **Section IV-A** (System Overview): One paragraph walking through the full pipeline step by step. Expand the Figure 1 caption to explicitly mention parallel execution, top-50 from each retriever, fusion to ~100, reranking to top-10.
- **Section IV-B** (Sparse + Dense): BM25 is already there. For BGE, you need to add: embedding dimensionality (1024), HNSW parameter justification (M=32, efConstruction=200), L2 normalization → cosine via inner product explanation.
- **Section IV-C** (RRF): Add the full formula, define all variables, justify k=60 citing Cormack et al. [8], explain what happens with smaller vs larger k.
- **Section IV-D** (ColBERT): Currently just the formula. Add the intuition paragraph, the efficiency argument (applied only to ~100 candidates), and token-level matching explanation.
- **Section IV-E** (Complexity): Add the stage-wise complexity table — BM25 O(log N), HNSW O(log N), RRF O(k), ColBERT O(m × n × candidates).
- **Section V** (Results): Currently placeholder rows. Fill in after experiments complete.
- **Section VI** (Discussion): Write failure case categories after running the failure analysis notebook.

---

### One Important Discrepancy to Resolve

Your paper abstract says **"Gemini Embedding 2"** and the title says "GemCol: Gemini Embeddings and ColBERTv2" — but your actual implementation uses **BGE-large-en-v1.5**, not Gemini. Your memory prompt, evaluation checklist, and all notebooks confirm BGE. You'll need to either update the title/abstract to match BGE, or actually implement the Gemini embedding version. This inconsistency will be caught in peer review.