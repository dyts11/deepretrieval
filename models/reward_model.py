"""
Dedicated Reward Model for RL-based Query Augmentation
Handles reward computation for generated queries using PubMed API
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
from utils.pubmed_api import PubmedAPI, create_pubmed_retriever
import asyncio
import math


def compute_ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    ideal_gains = [1.0] * min(k, len(relevant))
    ideal_dcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    if ideal_dcg == 0:
        return 0.0
    gains = [(1.0 if pmid in set(relevant) else 0.0) for pmid in retrieved[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    return max(0.0, min(1.0, dcg / ideal_dcg))


def compute_mrr_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    relevant_set = set(relevant)
    for idx, pmid in enumerate(retrieved[:k], start=1):
        if pmid in relevant_set:
            return 1.0 / idx
    return 0.0


class RetrievalRewardModel(nn.Module):
    """
    Reward model that computes retrieval-based rewards for generated queries.

    r = w_rec*recall@K + w_prec*precision@K + w_ndcg*ndcg@K + w_mrr*mrr@K + density_bonus
    All components are in [0,1].
    """

    def __init__(
        self,
        pubmed_api: PubmedAPI,
        top_k: int = 50,
        reward_scale: float = 5.0,
        min_reward: float = 0.0,
        max_reward: float = 5.0,
        # weights (retrieval-only by default)
        w_recall: float = 1,
        w_precision: float = 0,
        w_ndcg: float = 0,
        w_mrr: float = 0,
    ):
        super().__init__()
        self.pubmed_api = pubmed_api
        self.top_k = top_k
        self.reward_scale = reward_scale
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.weights = {
            'recall': w_recall,
            'precision': w_precision,
            'ndcg': w_ndcg,
            'mrr': w_mrr,
        }
        self.total_queries = 0
        self.total_reward = 0.0
        self.reward_history = []

    def forward(
        self,
        queries: List[str],
        relevant_pmids_list: List[List[str]],
        **kwargs
    ) -> torch.Tensor:
        rewards = []
        for query, relevant_pmids in zip(queries, relevant_pmids_list):
            reward = self._compute_single_reward(query, relevant_pmids)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)

    def _compute_single_reward(
        self,
        query: str,
        relevant_pmids: List[str]
    ) -> float:
        if not relevant_pmids:
            return self.min_reward
        try:
            # 1) Retrieve with the full query first
            retrieved_pmids = self.pubmed_api.search_with_keywords(query, topk=self.top_k)

            # 2) If full-query gets zero docs, try safer fallback using 2+ clause combinations
            #    We prefer OR-combinations of two clauses; if none reach a small threshold, fall back to best single clause.
            #if len(retrieved_pmids) == 0:
            #    import re as _re
            #    first_line = query.splitlines()[0].strip()
            #    parts = [_p.strip() for _p in _re.split(r"\bAND\b|\bOR\b", first_line, flags=_re.IGNORECASE) if _p.strip()]
            #    best_pmids = []
            #    # try all 2-clause OR combinations first
            #    for i in range(len(parts)):
            #        for j in range(i + 1, len(parts)):
            #            comb_query = f"{parts[i]} OR {parts[j]}"
            #            pmids_comb = self.pubmed_api.search_with_keywords(comb_query, topk=self.top_k)
            #            if len(pmids_comb) > len(best_pmids):
            #                best_pmids = pmids_comb
            #    retrieved_pmids = best_pmids
            #    # if still weak, try single parts and keep the best
            #    if len(retrieved_pmids)== 0:
            #        for part in parts:
            #            pmids_part = self.pubmed_api.search_with_keywords(part, topk=self.top_k)
            #            if len(pmids_part) > len(best_pmids):
            #                best_pmids = pmids_part
            #        if best_pmids:
            #            retrieved_pmids = best_pmids
            
            relevant_set = set(relevant_pmids)
            topk_list = retrieved_pmids[: self.top_k]
            retrieved_set = set(topk_list)

            recall = len(retrieved_set & relevant_set) / max(1, len(relevant_set))
            precision = len(retrieved_set & relevant_set) / max(1, len(retrieved_set))
            ndcg = compute_ndcg_at_k(topk_list, relevant_pmids, self.top_k)
            mrr = compute_mrr_at_k(topk_list, relevant_pmids, self.top_k)

            comp = {
                'recall': recall,
                'precision': precision,
                'ndcg': ndcg,
                'mrr': mrr,
            }
            #reward = sum(self.weights[k] * comp[k] for k in comp.keys())

            # 3) Retrieval density shaping: encourage queries that return a healthy number of docs
            density_target = max(10, self.top_k)  # aim for at least top_k hits
            density = min(1.0, len(retrieved_pmids) / float(density_target))
            #reward += 0.2 * density

            #reward = max(self.min_reward, min(self.max_reward, reward * self.reward_scale))
            q = query.split()
            length = len(q)
            reward = 2 - length/16
            #self._update_stats(reward, query, retrieved_pmids, relevant_pmids)
            return reward
        except Exception as e:
            print(e)
            # On error, return minimum reward (no format fallback)
            return self.min_reward

    #def _update_stats(
    #    self,
    #    reward: float,
    #    query: str,
    #    retrieved_pmids: List[str],
    #    relevant_pmids: List[str]
    #):
    #    self.total_queries += 1
    #    self.total_reward += reward
    #    self.reward_history.append({
    #        'reward': reward,
    #        'query': query[:100],
    #        'retrieved_count': len(retrieved_pmids),
    #        'relevant_count': len(relevant_pmids)
    #    })

    #def get_stats(self) -> Dict[str, Any]:
    #    if self.total_queries == 0:
    #        return {
    #            'avg_reward': 0.0,
    #            'total_queries': 0,
    #            'reward_history': []
    #        }
    #    return {
    #        'avg_reward': self.total_reward / self.total_queries,
    #        'total_queries': self.total_queries,
    #        'reward_history': self.reward_history[-10:]
    #    }

    #def reset_stats(self):
    #    self.total_queries = 0
    #    self.total_reward = 0.0
    #    self.reward_history = []

    def compute_reward(
        self,
        query: str,
        relevant_pmids: List[str]
    ) -> float:
        return self._compute_single_reward(query, relevant_pmids)

    def compute_rewards_batch(
        self,
        queries: List[str],
        relevant_pmids_list: List[List[str]]
    ) -> List[float]:
        if len(queries) != len(relevant_pmids_list):
            raise ValueError("Number of queries must match number of relevant PMID lists")
        results = [0.0] * len(queries)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def compute_single(idx, q, pmids):
            try:
                return idx, self._compute_single_reward(q, pmids)
            except Exception:
                return idx, self.min_reward

        max_workers = min(5, len(queries))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(compute_single, i, q, pmids) for i, (q, pmids) in enumerate(zip(queries, relevant_pmids_list))]
            for fut in as_completed(futures):
                idx, val = fut.result()
                results[idx] = val
        return results

    #async def compute_rewards_async(self, queries: List[str], relevant_pmids_list: List[List[str]]) -> List[float]:
    #    if len(queries) != len(relevant_pmids_list):
    #        raise ValueError("Number of queries must match number of relevant PMIDs lists")
    #    tasks = [self.pubmed_api.compute_reward_async(q, pmids) for q, pmids in zip(queries, relevant_pmids_list)]
    #    rewards = await asyncio.gather(*tasks, return_exceptions=True)
    #    cleaned = []
    #    for r in rewards:
    #        if isinstance(r, Exception):
    #            cleaned.append(self.min_reward)
    #        else:
    #            cleaned.append(r)
    #    return cleaned


def create_reward_model(
    api_key: Optional[str] = None,
    top_k: int = 50,
    reward_scale: float = 1.0,
    offline: bool = False,
    cache_path: Optional[str] = None,
    offline_corpus_path: Optional[str] = None,
) -> RetrievalRewardModel:
    """Factory to create a retrieval reward model with caching and optional offline mode."""
    pubmed_api = create_pubmed_retriever(
        api_key=api_key,
        offline=offline,
        cache_path=cache_path,
        offline_corpus_path=offline_corpus_path,
    )
    return RetrievalRewardModel(
        pubmed_api=pubmed_api,
        top_k=top_k,
        reward_scale=reward_scale,
    ) 