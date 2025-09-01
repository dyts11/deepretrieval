## Reward Design for DeepRetrieval PPO

This document describes how rewards are computed for training the policy to generate PubMed Boolean queries.

### Overview
- **Goal**: Optimize generated queries for retrieval quality while keeping outputs well‑formed, English, and concise.
- **Core idea**: Compute retrieval metrics on results from the PubMed retriever (with caching/offline support), add shaping to reduce early sparsity, and penalize off‑format/multilingual drift.

### Retrieval path
1. Retrieve with the **full generated query** first.
2. If full query retrieves **0 documents**:
   - Parse the first output line; split by `AND`/`OR` into clauses.
   - Try all 2‑clause `OR` combinations; if any returns at least a small threshold (≥ 10 docs), use that.
   - Otherwise, use the single clause that returns the most docs.
   - Apply a small penalty if fallback was used.

### Metrics (computed on top_k results)
Given:
- `retrieved_pmids` = PMIDs returned by the retriever
- `relevant_pmids` = gold PMIDs for the example
- `top_k` (default 50–100)

We compute:
- **Recall@K**: |retrieved ∩ relevant| / |relevant|
- **Precision@K**: |retrieved ∩ relevant| / K
- **nDCG@K**: DCG(retrieved vs relevant)/IDCG (binary gains)
- **MRR@K**: 1/rank of first relevant in top‑K, else 0
- **Efficiency**: A smooth penalty for overly long queries (shorter is better up to a cap)

Weighted sum (default weights):
- recall: 0.6
- precision: 0.05
- nDCG: 0.25
- MRR: 0.10
- efficiency: 0.0 (kept available)

### Density shaping (combat sparse zero‑rewards)
- Many full queries initially retrieve few or zero docs; overlap metrics become 0.
- We add a **density term** that rewards returning a healthy number of documents even before overlap appears:
  - `density_target = max(10, top_k)`
  - `density = min(1.0, len(retrieved_pmids) / density_target)`
  - Add `0.2 * density` to the reward.

This provides a non‑zero gradient toward “retrieve more,” which increases the chance of future overlaps.

### Penalties and guardrails
- **Boolean format**: If the extracted query lacks `AND`/`OR`/`NOT`, multiply reward by 0.7.
- **English bias**: If ASCII character ratio < 0.8, multiply reward by 0.5 to reduce multilingual drift.
- **Fallback penalty**: If subquery fallback was used, multiply reward by 0.7.

These discourage off‑format and multilingual outputs while retaining some learning signal.

### Scaling and clamping
- Reward is clamped to `[min_reward, max_reward]` and then multiplied by `reward_scale`:
  - Defaults: `min_reward=0.0`, `max_reward=1.0`, `reward_scale=1.0`.
  - If you want larger magnitudes, increase both `reward_scale` and `max_reward` together (e.g., 5.0 each). Note PPO advantage whitening limits the effect of simple scaling.

### Pseudocode
```python
retrieved_pmids = search(full_query, top_k)
if len(retrieved_pmids) == 0:
    parts = split_by_and_or(first_line(full_query))
    best = []
    # try 2-clause OR combos
    for (a, b) in combinations(parts, 2):
        pmids_comb = search(f"{a} OR {b}", top_k)
        if len(pmids_comb) >= 10:   # threshold_docs
            retrieved_pmids = pmids_comb
            used_fallback = True
            break
        best = argmax_by_len([best, pmids_comb])
    if len(retrieved_pmids) == 0 and best:
        retrieved_pmids = best
        used_fallback = True

# metrics (R, P, nDCG, MRR) on top_k_list = retrieved_pmids[:top_k]
reward = 0.6*R + 0.05*P + 0.25*nDCG + 0.10*MRR + 0.0*efficiency

# density shaping
density_target = max(10, top_k)
density = min(1.0, len(retrieved_pmids) / density_target)
reward += 0.2 * density

# penalties
if no_boolean_ops(query):
    reward *= 0.7
if ascii_ratio(query) < 0.8:
    reward *= 0.5
if used_fallback:
    reward *= 0.7

# clamp & scale
reward = clamp(reward, min_reward, max_reward) * reward_scale
```

### Key parameters (current defaults)
- `top_k`: 50–100 (use 100 to reduce early sparsity)
- `reward_scale`: 1.0 (with `max_reward=1.0`)
- `threshold_docs` for fallback: 10
- Density weight: 0.2
- Penalties: boolean‑missing (×0.7), non‑ASCII (×0.5), fallback (×0.7)

### Practical tips
- Keep `top_k` high (e.g., 100) for denser early signal.
- Tighten KL and use lower sampling temperature during training generation to prevent drift while rewards are still sparse.
- Consider logging the proportion of fallbacks and ascii_ratio distributions to monitor stability.

### Where it lives in code
- Implementation: `models/reward_model.py` in `_compute_single_reward`.
- PubMed retriever and caching: `utils/pubmed_api.py`.
- Test hooks: see `training/full_training.py` for usage in PPO. 