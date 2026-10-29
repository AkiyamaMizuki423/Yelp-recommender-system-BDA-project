# Final Project Report: Multi-Stage Offline Ranking System for Yelp Restaurant Discovery

## 1. Executive Summary

This report documents the final implementation of our big data course project, which evolved from an initial recommendation prototype into a three-layer offline ranking system for Yelp restaurant discovery. The final system is designed to improve recommendation quality in a controlled and explainable way by combining large-scale candidate generation, structured machine learning reranking, and bounded LLM-style reward-model rescue reranking.

The final project is not only a code deliverable. It is also an implementation journal that records how the architecture matured, what technical trade-offs we faced, which tools we explored, and why the final design has business value. Compared with the original proposal, the final version provides a more complete description of the system architecture, modeling decisions, evaluation results, engineering workflow, and deployment considerations.

Our final architecture contains three core stages:

1. Stage09: recall routing and candidate fusion
2. Stage10: structured reranking with feature-rich machine learning models
3. Stage11: bounded reward-model rescue reranking based on segmented experts

The overall idea is simple: do not ask a large model to rank everything. Instead, use classical large-scale recommendation methods to narrow the candidate set first, then apply stronger models only where they add the most value. This design improves quality while keeping cost, latency, and risk under control.

## 2. Project Background and Problem Statement

Restaurant discovery platforms face a classic recommendation problem: users have many choices, but only a small number of those choices truly match their preferences, context, and recent intent. A good recommender must do more than surface popular items. It must identify the right restaurant for the right user at the right time.

In practice, this is difficult for several reasons:

- the candidate space is large
- user intent is noisy and dynamic
- historical preference alone is not enough
- strong candidates may be under-ranked by global models
- full-list semantic reranking is expensive and hard to control

Our project addresses this problem by building an offline ranking stack for Yelp restaurant recommendation. The system focuses on restaurant discovery and ranking quality rather than review generation or chatbot interaction. The core task is: given a user and a candidate set of restaurants, rank the restaurants so that the most relevant ones appear in the top positions.

## 3. Objectives

The final implementation pursued five objectives:

1. Build a reproducible large-scale offline recommendation pipeline on Yelp-style data.
2. Improve recall before ranking by combining multiple candidate sources instead of relying on a single retrieval strategy.
3. Build a stronger structured reranker using match, semantic, competition-aware, and route-derived features.
4. Explore whether LLM-style reasoning can improve ranking quality when used in a bounded and carefully controlled way.
5. Produce a final system with clear business value, interpretable architecture, and realistic deployment logic.

## 4. Dataset Scope and Experimental Setting

The repository is organized around Yelp restaurant discovery. The current public project line uses the following filtered scale after preprocessing:

- 196,939 users
- 1,798 businesses

To evaluate the system under different interaction densities, we defined three user-set lines:

- bucket2: cold-start-inclusive trainable set under leave-two-out
- bucket5: mid-to-high interaction set
- bucket10: high-interaction set

The current public scope is centered on bucket5 because it provides a practical balance between strong signal quality and broad enough user coverage.

Current bucket5 scope:

- 1,798 businesses
- 9,765 users
- 133,048 train interactions
- 9,765 validation users
- 9,765 test users
- 1,935 fixed Stage10 evaluation users
- 517 Stage11 rescue evaluation users

This bucket-based design was important during implementation. We first validated the architecture on the cleaner high-interaction bucket10 line, then expanded to bucket5 to confirm that the same design still worked under a broader and more practical data regime.

## 5. End-to-End System Architecture

The final system is a layered offline ranking stack.

### 5.1 High-Level Pipeline

1. Data preparation and historical foundation stages (Stage01 to Stage08)
2. Stage09 recall routing and candidate fusion
3. Stage10 structured reranking
4. Stage11 reward-model rescue reranking on a bounded candidate window
5. Offline evaluation using Recall@10 and NDCG@10

### 5.2 Why a Layered Architecture Was Chosen

A layered design was chosen instead of a single end-to-end model for both engineering and modeling reasons:

- large candidate pools require scalable retrieval logic
- structured models are strong at global ranking and feature fusion
- semantic or LLM-based models are more useful for local, difficult comparisons
- bounded reranking reduces inference cost and protects top-rank stability
- each stage can be diagnosed, ablated, and improved independently

This architecture also matches a realistic product setting. In industry systems, candidate generation, mainline ranking, and final rescue or reordering are often handled separately because each layer solves a different part of the problem.

## 6. System Components in Detail

### 6.1 Stage01 to Stage08: Historical Foundation

Although the current reproducible release path begins at Stage09, earlier stages remain important because they record the project's evolution.

These stages include:

- data preparation
- exploratory analysis
- ALS-based recommendation experiments
- cluster-based business profiling
- embedding-based semantic grouping
- relabeling and refinement workflows

This earlier work helped us understand user behavior, business categories, and the kinds of semantic signals that later became useful in Stage09 to Stage11.

An especially useful lesson from the early stages was that recommendation quality could not be improved by collaborative filtering alone. Semantic signals, cluster structure, and user-intent evidence needed to be integrated into the pipeline.

### 6.2 Stage09: Recall Routing and Candidate Fusion

Stage09 is responsible for building the candidate pool. This layer combines multiple sources instead of trusting a single retrieval channel.

The active candidate sources include signals such as:

- ALS-style collaborative filtering
- cluster/profile-based retrieval
- popularity-based retrieval
- route-aware contextual candidates

The main role of Stage09 is not final ranking. Its role is to preserve promising candidates, reduce hard misses, and deliver a high-quality bounded pool to downstream models.

This stage uses route-aware candidate routing rather than plain candidate merging. That means different sources are not treated as equally meaningful in every situation. The system controls recall budgets, candidate lanes, and challenger lanes so that the final fused pool contains both strong head candidates and useful alternatives.

This design matters because a ranking model cannot recover a restaurant that never appears in the candidate set. Therefore, Stage09 acts as the recall safety net of the system.

#### Stage09 Result

For bucket5, the candidate funnel improved from an early gate to the current structural v5 source-parity line:

- truth_in_pretrim150 improved from 0.7248 to 0.7451
- hard_miss dropped from 0.1616 to 0.1190

This shows that recall routing materially improved the chance that the true item survived into the later ranking stages.

### 6.3 Stage10: Structured Reranking

Stage10 is the global ranking backbone of the system. It receives the candidate pool from Stage09 and assigns more precise ordering using structured features and learned reranking models.

The Stage10 design combines multiple feature families:

- route and source features
- user-business match features
- text-match and semantic similarity features
- relative cross features
- group-gap and competition-aware features
- rank priors and calibration features

The implementation includes both conventional machine learning models and a more expressive wide-and-deep style component. In practical terms, Stage10 is the part of the system that learns how to balance historical fit, semantic alignment, source quality, and local competition.

Why this stage is important:

- it creates a stable mainline ranking before any LLM-style intervention
- it reduces noise in the candidate list
- it provides a strong baseline that can be trusted on its own
- it creates the bounded window used by Stage11

#### Stage10 Result

Current fixed evaluation results show consistent gains over a prescore baseline:

- bucket2: Recall@10 improved from 0.1098 to 0.1127; NDCG@10 improved from 0.0513 to 0.0522
- bucket5: Recall@10 improved from 0.0935 to 0.1261; NDCG@10 improved from 0.0440 to 0.0581
- bucket10: Recall@10 improved from 0.0569 to 0.0772; NDCG@10 improved from 0.0265 to 0.0341

The improvement on bucket5 is especially meaningful because it is the main outward-facing use case.

### 6.4 Stage11: Reward-Model Rescue Reranking

Stage11 is the most innovative part of the project. Instead of asking a large model to rerank the full candidate list, we apply reward-model rescue reranking only to the bounded window already produced by Stage10.

This choice was deliberate.

A full-list LLM reranker would be:

- expensive
- harder to control
- more likely to destabilize top ranks
- more difficult to roll back in production

Our system avoids that risk by using Stage11 only as a bounded rescue layer.

#### Core Idea

Stage11 focuses on candidates that are already plausible but may still be under-ranked. The reward model is used to answer local questions such as:

- should this boundary candidate be promoted above nearby blockers?
- should this mid-rank candidate enter the top positions?
- should this deep-rank candidate receive a conservative uplift?

This is not generic text generation. It is a ranking-oriented reward-model design.

#### Segmented Expert Design

The final Stage11 line uses one shared reward-model backbone and three specialized experts:

- 11-30 boundary rescue expert
- 31-60 mid-rank rescue expert
- 61-100 deep-rank uplift expert

This segmentation was introduced because different rank regions behave differently.

- candidates in positions 11-30 are close to the top and require careful local comparison
- candidates in positions 31-60 need stronger rescue logic to overcome nearby blockers
- candidates in positions 61-100 are much deeper, so aggressive promotion can be risky

The implementation therefore uses specialized experts and bounded policy gates rather than one uniform rescue model.

#### Why This Is Creative and Innovative

The creative contribution of this project is not simply ¡°using an LLM.¡± The innovation lies in how the model is used:

- the LLM-style component does not replace the main ranker
- the model is used only where semantic judgment is most useful
- reward modeling is segmented by rerank window
- local listwise prompts are built from visible evidence rather than hidden labels
- policy gates protect top-rank stability and control promotion behavior

This makes the design both innovative and practically grounded.

#### Stage11 Result

Current reference points on bucket5:

- v120 two-band line at alpha = 0.80: Recall@10 = 0.1973, NDCG@10 = 0.0898
- v124 tri-band frozen line at alpha = 0.36: Recall@10 = 0.1857, NDCG@10 = 0.0838

These results show that bounded semantic rescue can provide substantial uplift beyond the structured reranker alone.

## 7. Model Explanation

### 7.1 Retrieval and Routing Logic

At the retrieval stage, the system gathers candidates from multiple recommendation channels. This is closer to a heterogeneous recall framework than to a single recommender. Different sources contribute complementary strengths:

- collaborative filtering captures co-consumption patterns
- cluster/profile retrieval captures category or taste neighborhoods
- popularity provides robust fallback coverage
- context-aware routes add challenge candidates that might otherwise be missed

Candidate fusion then converts these heterogeneous sources into a shared ranking surface.

### 7.2 Stage10 Machine Learning Model Logic

Stage10 uses structured feature engineering and learned reranking. The project includes a wide-and-deep style module and tree-based ranking/classification components.

Conceptually:

- the wide side captures useful memorized crosses such as user segment x category or user segment x business
- the deep side helps integrate dense and continuous signals
- ranking-oriented sampling and calibration make the model sensitive to useful top-rank errors

The model is trained not only to score candidates, but also to behave well under ranking evaluation metrics such as Recall@10 and NDCG@10.

### 7.3 Stage11 Reward Model Logic

The reward model is trained on pairwise or listwise local comparison tasks. Rather than learning from the entire ranked list at once, the model learns to compare a focus candidate with a small rescue slate of nearby rivals.

A typical prompt contains:

- user focus and recent intent
- avoidance cues and history pattern
- candidate business evidence
- local rival information
- the candidate's current rank window

Importantly, the hidden truth label is not revealed at inference time. This design reduces leakage and makes the model behave more like a realistic post-ranker.

The model judges whether a candidate is under-ranked relative to local competitors, then produces a rescue signal that is fused with the Stage10 baseline.

## 8. Implementation Experience and Development Journal

This final report also serves as a journal of implementation experience. Several major lessons emerged during the project.

### 8.1 Early Lesson: One Model Could Not Solve Everything

In the early stages, we explored collaborative filtering, baseline reranking, relabeling, clustering, and semantic representations. These experiments were useful, but they also showed that recommendation quality did not improve reliably when we depended on only one technique.

This motivated the layered architecture.

### 8.2 Recall Quality Was More Important Than Expected

One major lesson was that many downstream errors were not ranking errors at all. In some cases, the true item was already missing before the ranker even started. That made Stage09 recall routing a critical component of the final design.

### 8.3 Structured Features Still Matter a Lot

Even though modern LLM-style methods are powerful, the project showed that strong structured reranking remains essential. The biggest gains came from combining route-aware features, text-match features, relative comparison features, and group-gap signals into a stable global reranker.

### 8.4 LLMs Were Most Useful for Local Difficult Cases

Another important lesson was that LLM-style reasoning is most valuable when applied to ambiguous local comparisons rather than to full-list ranking. In other words, semantic judgment worked best as a rescue mechanism, not as a total replacement for the recommendation stack.

### 8.5 Segmented Experts Worked Better Than a Single Rescue Policy

A single rescue policy across the entire rank range was not ideal. Boundary rescue, mid-rank rescue, and deep-rank uplift behave differently. Splitting the problem into multiple experts improved stability and made the final logic easier to interpret.

### 8.6 Engineering Reproducibility Matters

As the project became larger, launcher wrappers, path contracts, bucket-specific scripts, and structured runtime conventions became necessary. This was an important big-data engineering lesson: a model is only useful if the experiment path is reproducible and understandable.

## 9. Tools and Technologies Used

The project combines classic data engineering tools with modern machine learning tooling.

### 9.1 Core Tools

- Python for data processing and modeling
- Pandas and NumPy for tabular processing
- PySpark for large-scale pipeline execution
- Scikit-learn for classical ML utilities
- XGBoost for strong structured reranking baselines
- PyTorch for neural ranking and reward-model training
- Hugging Face Transformers for Stage11 model loading and training
- PEFT and LoRA for parameter-efficient fine-tuning
- BitsAndBytes for 4-bit QLoRA-style GPU-efficient training

### 9.2 Additional Tools Explored During the Project

The project also explored extra tools beyond the minimal baseline:

- SentenceTransformers for semantic embedding generation
- BAAI BGE-M3 and MiniLM embedding models for representation learning
- Ollama for local LLM-assisted taxonomy and relabeling experiments
- Qwen family models as the backbone for Stage11 reward-model experiments

These tools were not all used in exactly the same way in the final frozen line, but they were important in the project's exploration, ablation, and iterative refinement.

## 10. Installation Instructions

This section documents how to reproduce or extend the environment, including additional tools explored during the project.

### 10.1 Basic Python Environment

Recommended environment:

- Python 3.10 or above
- virtual environment via `venv` or Conda

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The basic requirements file covers the CPU-friendly baseline used for smoke tests and non-GPU stages.

Main baseline packages include:

- numpy
- pandas
- pyarrow
- scipy
- scikit-learn
- xgboost
- pyspark
- pytest

### 10.2 PySpark and Java

PySpark requires a working Java runtime.

Recommended steps:

1. Install JDK 17 or a compatible Java version.
2. Set `JAVA_HOME` to the JDK installation path.
3. Add Java `bin` to `PATH`.
4. Install Python dependencies with `pip install -r requirements.txt`.

Verification:

```bash
java -version
python -c "import pyspark; print('pyspark ok')"
```

### 10.3 GPU / Stage11 Reward-Model Environment

For GPU-oriented Stage11 experiments, install the additional requirements:

```bash
pip install -r requirements-stage11-qlora.txt
```

This adds packages such as:

- torch
- transformers
- accelerate
- peft
- datasets
- sentencepiece
- safetensors
- bitsandbytes

Recommended notes:

- use a CUDA-compatible GPU for efficient Stage11 training
- ensure the installed PyTorch build matches the CUDA version
- `bitsandbytes` is recommended for 4-bit QLoRA runs
- if no GPU is available, Stage11 training becomes much slower and may be impractical

### 10.4 Hugging Face Model Access

Some Stage11 runs require access to pretrained model checkpoints from Hugging Face.

Recommended setup:

```bash
pip install huggingface_hub
huggingface-cli login
```

If running in a restricted environment, models can also be cached locally and referenced through environment variables.

### 10.5 Ollama for Local LLM Experiments

The project explored local LLM-assisted relabeling using Ollama.

Installation steps:

1. Install Ollama from the official distribution for your OS.
2. Start the Ollama service.
3. Pull a model, for example:

```bash
ollama pull qwen3:8b
```

4. Verify the local API is available at:

```text
http://localhost:11434
```

This tool was useful for exploratory labeling workflows in the earlier stages.

### 10.6 SentenceTransformers for Embedding Experiments

For semantic embedding and clustering experiments, install:

```bash
pip install sentence-transformers
```

Optional models explored in the project include:

- `BAAI/bge-m3`
- `sentence-transformers/all-MiniLM-L6-v2`

These models were used in embedding-based clustering and semantic representation experiments.

## 11. Evaluation Metrics

The main evaluation metrics are:

- Recall@10
- NDCG@10

These metrics are suitable because the task is top-rank recommendation rather than full-list regression. The goal is to ensure that relevant restaurants appear near the top of the ranked list.

We also used stage-specific diagnostics such as:

- truth retention inside pretrim windows
- hard miss rate
- win rates in Stage11 expert training and evaluation

These intermediate diagnostics were important because they helped us locate where gains or failures were happening.

## 12. Creativity and Innovation in the Proposed Work

This project demonstrates creativity and innovation in several ways.

### 12.1 Bounded LLM Usage Instead of Naive End-to-End Replacement

A common trend is to use LLMs everywhere. Our project took a more disciplined approach. We used the LLM-style component only where it could add meaningful value, namely local rescue reranking. This is more practical and more innovative than simply replacing everything with a large model.

### 12.2 Segmented Expert Reward Modeling

The segmented Stage11 design is a novel contribution in the project context. Rather than training one generic rescue model, we built specialized experts for different rerank windows and fused them with bounded policies.

### 12.3 Integration of Classical and Modern Methods

The project combines classical recommender-system logic, feature engineering, tree models, wide-and-deep learning, semantic representations, and reward-model reasoning in one coherent stack.

### 12.4 Reproducible Big-Data Engineering

Innovation is not only about algorithms. The project also improved reproducibility through launcher wrappers, variable contracts, bounded run surfaces, and modular stage organization.

## 13. Business Value

The final system has clear business value for a restaurant discovery platform.

### 13.1 Better User Experience

Improved top-rank recommendation quality means users are more likely to find relevant restaurants quickly. This can improve:

- click-through rate
- conversion to orders or visits
- session satisfaction
- user retention

### 13.2 Better Use of Semantic Intent

The Stage11 rescue layer helps identify cases where a restaurant is semantically right for a user but is under-ranked by a purely structured model. This is valuable in competitive recommendation spaces where many candidates appear superficially similar.

### 13.3 Lower Operational Risk

Because the LLM-style logic is bounded to a candidate window rather than the full list, the system is easier to control, cheaper to run, and safer to deploy gradually.

### 13.4 Clear Upgrade Path

The modular architecture supports incremental improvement. The business can upgrade recall, reranking, or rescue logic independently rather than rebuilding the entire pipeline.

## 14. Limitations

Despite the strong results, the project still has limitations.

- evaluation is offline rather than online A/B testing
- Stage11 GPU training can be resource-intensive
- deep-rank rescue is still conservative in the current freeze line
- some exploratory semantic tools were not fully standardized into one production path
- business metrics such as CTR or revenue lift were inferred from ranking improvements rather than measured directly

These limitations are normal for a course project, but they also define useful future work.

## 15. Future Work

Possible next steps include:

1. online-style simulation or A/B-style evaluation
2. stronger cold-start handling for sparse users and new businesses
3. better calibration of the 61-100 expert for safe promotion
4. richer temporal and location-aware intent features
5. more efficient inference and caching for Stage11 prompts and scores
6. automated monitoring dashboards for stage-by-stage failure analysis

## 16. Conclusion

This project successfully evolved from an initial recommendation proposal into a comprehensive multi-stage offline ranking system for Yelp restaurant discovery. The final contribution is not just a collection of scripts, but a coherent architecture that combines large-scale retrieval, structured reranking, and bounded reward-model rescue.

The most important lesson from the implementation is that recommendation quality improves when each model is used for the part of the problem it solves best. Stage09 protects recall, Stage10 provides a reliable global backbone, and Stage11 adds semantic rescue only where local difficult comparisons justify the extra complexity.

From both an academic and practical perspective, the project demonstrates creativity, engineering maturity, and meaningful business relevance. It also shows that modern LLM-style methods can be integrated into recommender systems in a controlled and value-adding way rather than as an unbounded replacement for traditional ranking pipelines.

## 17. Appendix: Key Final Results Snapshot

### Stage09 Recall Routing

- bucket5 truth_in_pretrim150: 0.7451
- bucket5 hard_miss: 0.1190

### Stage10 Structured Reranking

- bucket2 LearnedBlendXGBCls@10: Recall 0.1127, NDCG 0.0522
- bucket5 LearnedBlendXGBCls@10: Recall 0.1261, NDCG 0.0581
- bucket10 LearnedBlendXGBCls@10: Recall 0.0772, NDCG 0.0341

### Stage11 Rescue Reranking

- two-band v120 @ alpha 0.80: Recall@10 0.1973, NDCG@10 0.0898
- tri-band frozen v124 @ alpha 0.36: Recall@10 0.1857, NDCG@10 0.0838
