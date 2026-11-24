# YZM5109 – Assignment II, Option C: Naive RAG vs. Improved RAG

**Research question.** Can retrieval or chunking improvements outperform a naive RAG baseline?

| Condition | Chunking | Retrieval | Role |
|---|---|---|---|
| **C1** | fixed-size, 256 tokens, 32-token overlap | dense (bge-small + FAISS) | naive baseline |
| **C2** | same as C1 | **hybrid**: dense top-50 + BM25 top-50, Reciprocal Rank Fusion (k = 60) | improvement 1 (retrieval) |
| **C3** | **semantic** (cut at topic changes, percentile 80 chosen on dev, 32–256 tokens) | dense | improvement 2 (chunking) |
| **C4** | semantic | hybrid | optional combination |
| C5 | semantic, minimum 256 tokens (size close to C1) | dense | **post-hoc** check, added after seeing C1–C4 |

Everything else is identical across conditions: corpus, 330 test questions, embedding model, top-5 context,
prompt, generator (Qwen2.5-3B-Instruct, greedy) and evaluation.

## Main results (test set: 300 answerable + 30 unanswerable questions)

| Cond. | Recall@5 | MRR@10 | Strict Hit@5 | F1 | Answer recall | Faithful (3-sent. windows) | Abstention acc. | LLM-judge correct* |
|---|---|---|---|---|---|---|---|---|
| C1 | 0.768 | 0.644 | 0.797 | 0.403 | 0.617 | 0.776 | 0.400 | 0.82 |
| C2 | **0.890** | 0.793 | **0.910** | **0.465** | **0.705** | **0.799** | 0.367 | **0.88** |
| C3 | 0.739 | 0.709 | 0.733 | 0.409 | 0.612 | 0.746 | 0.333 | 0.80 |
| C4 | 0.847 | **0.803** | 0.847 | 0.452 | 0.672 | 0.735 | **0.433** | 0.87 |
| C5 | 0.777 | 0.667 | 0.777 | 0.395 | 0.601 | 0.750 | 0.433 | – |

\*Gemini 3.5 Flash Lite on the same 100 answerable questions for C1–C4. A manual check of 20 random judge decisions
agreed on 17/20 correctness and 20/20 support labels (`results/judge_agreement.txt`).
Full table: `results/final_metrics_test.csv`; significance tests: `results/final_bootstrap_test.csv`.

**Summary.** Hybrid retrieval (C2) is the only robust improvement (retrieval +11–15 points, answer recall +8.8,
paired-bootstrap CIs exclude 0). Semantic chunking (C3) gives no robust gain: its MRR and chunk-level NLI gains
disappear under the strict-relevance and equal-length-window checks. It uses 35% less context for the same
correctness. C4 does not beat C2. Abstention on unanswerable questions is low (33–43%) for every condition.

## Folder structure

```
report.pdf                 IEEE report (3–4 pages + references + prompt appendix)
README.md                  this file
AI_usage_log.docx          AI usage log
code/                      Kaggle notebooks, run in the order below (saved with their outputs)
configs/                   all settings and prompts (JSON)
data/                      test/dev questions with gold evidence offsets, data statistics, unanswerable review log
results/                   per-question predictions, retrieved chunk IDs/ranks, metrics, bootstrap, judge files
screenshots/               metric screenshots for every condition
```

## How to run (Kaggle, GPU T4, Internet on)

Each notebook takes the **output of the previous notebook** as its input
(*Add Input → Notebooks → Your Work*) and copies its files forward, so the last notebook contains everything.

| Order | Notebook | Input | What it does |
|---|---|---|---|
| 1 | `01-rag-data-preparation.ipynb` | – | Loads RepLiQA `repliqa_4`, builds corpus (3,590 docs), locates gold evidence offsets, samples test (300+30) and dev (50+10) questions, BM25 review of unanswerable candidates |
| 2 | `02-rag-chunking-retrieval.ipynb` | NB 01 | Fixed and semantic chunking (percentile chosen on **dev**), dense and hybrid retrieval for C1–C4, Recall@k / MRR / bootstrap |
| 3 | `03-rag-generation.ipynb` | NB 02 | Dev check of the prompt, then Qwen2.5-3B answers for 330 test questions × C1–C4 |
| 4 | `05-rag-c5-size-matched.ipynb` | NB 03 | Post-hoc C5: size-matched semantic chunks, retrieval and generation |
| 5 | `04-rag-evaluation-and-llm-as-a-judge.ipynb` | NB 05 | Strict retrieval check, F1 / answer recall, NLI faithfulness (+ window check), abstention, bootstrap, error analysis, LLM judge, ZIP |

Measured times on one T4: generation 5.2–5.3 min per condition for fixed chunks and 3.8 min for semantic chunks
(C5: about 5 min); the LLM judge about 45 min for 400 requests.

Notebook 05 was added after Notebook 04 was designed, which is why it runs before Notebook 04.

**LLM judge.** Add a Gemini API key in *Add-ons → Secrets* as `GEMINI_API_KEY` (the key is never written in the
code). The free tier allows 500 requests/day for Gemini 3.5 Flash Lite; the judge uses 400. If the quota stops it,
re-run the cell: finished judgements in `results/judge_test.jsonl` are skipped.

**Not included (large, fully reproducible):** `corpus.jsonl` and the chunk files (~100 MB); Notebooks 01, 02
and 05 recreate them.

## Models, data and settings

| Item | Value |
|---|---|
| Dataset | `ServiceNow/repliqa`, split `repliqa_4` (all documents with ≥ 1,000 characters) |
| Embedding model | `BAAI/bge-small-en-v1.5` (commit `5c38ec7c`), query instruction as recommended by BAAI |
| Generator | `Qwen/Qwen2.5-3B-Instruct` (commit `aa8e7253`), fp16, greedy, max 64 new tokens, batch 8 |
| NLI model | `cross-encoder/nli-deberta-v3-base` |
| LLM judge | `gemini-3.5-flash-lite`, temperature 0, JSON output, 100 questions × C1–C4 |
| Top-k | 5 chunks in the prompt; 10 retrieved for metrics |
| Seed | 42 everywhere |
| Hardware | Kaggle, 1 × Tesla T4 |

Prompts: `configs/generation_config.json` (system prompt, user template, abstention regex) and
`configs/evaluation_config.json` (judge prompt). They are also printed in the report appendix.

## Key files

| File | Content |
|---|---|
| `data/test_questions.jsonl` | 330 test questions: question, reference answer, document ID, gold-evidence character offsets, answerable flag |
| `data/unanswerable_review.csv` | top BM25 passages from other documents for each unanswerable candidate (manual review log; 61 reviewed, 27 rejected) |
| `results/retrieval_test_C*.jsonl` | per question: top-10 retrieved chunk IDs, document IDs, ranks, scores, gold chunk IDs |
| `results/generation_test_C*.csv/.jsonl` | **predictions**: sample ID, question, gold answer, context chunk IDs, output, abstention flag |
| `results/final_metrics_test.csv` | final metric table for C1–C5 |
| `results/final_bootstrap_test.csv`, `retrieval_bootstrap_test*.csv` | paired bootstrap differences with 95% CIs |
| `results/retrieval_strict_test.csv` | strict-relevance robustness check |
| `results/error_types_test.csv`, `error_examples_test.csv` | retrieval miss vs. generation error analysis |
| `results/judge_test.jsonl`, `judge_metrics_test.csv`, `judge_bootstrap_test.csv` | LLM-judge labels and summaries |
| `results/judge_manual_check_filled.csv`, `judge_agreement.txt` | manual check of 20 judge decisions by the author |
| `results/dev_semantic_sweep.csv` | dev-set choice of the semantic-chunking percentile |
| `results/c5_size_matching.csv` | C5 size/boundary trade-off table |
