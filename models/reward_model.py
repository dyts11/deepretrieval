"""
Dedicated Reward Model for RL-based Query Augmentation
Handles reward computation for generated queries using PubMed API
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Set
from utils.pubmed_api import PubmedAPI, create_pubmed_retriever
import asyncio
import math
import re


def parse_boolean_query(query: str) -> List[str]:
    """
    Parse a Boolean query into individual search terms for retrieval.
    
    For a query like: ((Total Knee Arthroplasty Trial OR Total Knee Arthroplasty Surgery) AND (Drainage OR Antiotics Trial))
    
    This function will:
    1. Parse the Boolean structure
    2. Generate individual search terms based on AND/OR logic
    3. Return a list of search queries to execute
    
    Args:
        query: Boolean query string
        
    Returns:
        List of individual search queries to execute
    """
    if not query or not query.strip():
        return [query]
    
    # Clean and normalize the query
    query = query.strip()
    
    # Simple heuristic: if no Boolean operators, return as-is
    if not any(op in query.upper() for op in [' AND ', ' OR ', ' NOT ']):
        return [query]
    
    try:
        # Handle nested parentheses by expanding combinations
        # For complex queries, we'll use a simplified approach:
        # 1. Split on main AND operators
        # 2. For each AND clause, handle OR combinations
        # 3. Generate Cartesian product of combinations
        
        # Remove outer parentheses if they wrap the entire query
        query = query.strip()
        if query.startswith('(') and query.endswith(')'):
            # Check if these are the outermost parentheses
            paren_count = 0
            for i, char in enumerate(query):
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0 and i < len(query) - 1:
                        # Not outermost parentheses
                        break
            else:
                # These are outermost parentheses, remove them
                query = query[1:-1].strip()
        
        # Split on AND (case insensitive, but preserve original case in terms)
        and_parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)
        
        if len(and_parts) == 1:
            # No AND operators, might have OR operators
            or_parts = re.split(r'\s+OR\s+', query, flags=re.IGNORECASE)
            # Clean each part
            cleaned_parts = []
            for part in or_parts:
                cleaned = part.strip().strip('()')
                if cleaned:
                    cleaned_parts.append(cleaned)
            return cleaned_parts if cleaned_parts else [query]
        
        # Handle AND combinations
        # For each AND part, extract OR alternatives
        and_groups = []
        for and_part in and_parts:
            and_part = and_part.strip().strip('()')
            if ' OR ' in and_part.upper():
                # Split on OR
                or_alternatives = re.split(r'\s+OR\s+', and_part, flags=re.IGNORECASE)
                cleaned_alternatives = []
                for alt in or_alternatives:
                    cleaned = alt.strip().strip('()')
                    if cleaned:
                        cleaned_alternatives.append(cleaned)
                and_groups.append(cleaned_alternatives if cleaned_alternatives else [and_part])
            else:
                # Single term
                cleaned = and_part.strip()
                if cleaned:
                    and_groups.append([cleaned])
        
        if not and_groups:
            return [query]
        
        # Generate Cartesian product of AND groups
        import itertools
        combinations = list(itertools.product(*and_groups))
        
        # Create search queries from combinations
        search_queries = []
        for combo in combinations:
            # Join terms with AND for this combination
            combined_query = ' AND '.join(f'({term})' if ' ' in term else term for term in combo)
            search_queries.append(combined_query)
        
        # Also add individual terms for broader coverage
        all_terms = []
        for group in and_groups:
            all_terms.extend(group)
        
        # Add individual high-value terms
        for term in all_terms:
            if len(term.split()) >= 2:  # Multi-word terms are more specific
                search_queries.append(term)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in search_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        return unique_queries if unique_queries else [query]
        
    except Exception as e:
        # If parsing fails, fall back to original query
        print(f"Boolean query parsing failed for '{query}': {e}")
        return [query]


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
        top_k: int = 100,
        reward_scale: float = 5.0,
        min_reward: float = 0.0,
        max_reward: float = 5.0,
        # weights (retrieval-only by default)
        w_recall: float = 0.8,
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
        try:
            # 1) Parse Boolean query into multiple search queries
            search_queries = parse_boolean_query(query)
            
            # 2) Execute searches and combine results
            all_retrieved_pmids = []
            pmid_scores = {}  # Track how many queries retrieved each PMID
            
            for search_query in search_queries:
                try:
                    pmids = self.pubmed_api.search_with_keywords(search_query, topk=self.top_k)
                    for pmid in pmids:
                        if pmid not in pmid_scores:
                            pmid_scores[pmid] = 0
                            all_retrieved_pmids.append(pmid)
                        pmid_scores[pmid] += 1
                except Exception as e:
                    print(f"Search failed for query '{search_query}': {e}")
                    continue
            
            # 3) Sort by relevance score (number of queries that retrieved this PMID)
            # PMIDs retrieved by more queries are likely more relevant
            all_retrieved_pmids.sort(key=lambda pmid: pmid_scores[pmid], reverse=True)
            
            # 4) Take top-k results
            retrieved_pmids = all_retrieved_pmids[:self.top_k]
            
            # 5) Fallback: if no results from Boolean parsing, try original query
            if not retrieved_pmids:
                retrieved_pmids = self.pubmed_api.search_with_keywords(query, topk=self.top_k)
            
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
            reward = sum(self.weights[k] * comp[k] for k in comp.keys())

            # 3) Retrieval density shaping: encourage queries that return a healthy number of docs
            density_target = max(10, self.top_k)  # aim for at least top_k hits
            density = min(1.0, len(retrieved_pmids) / float(density_target))
            #reward += 0.2 * density

            #reward = max(self.min_reward, min(self.max_reward, reward * self.reward_scale))
            q = query.split()
            length = len(q)
            #reward = (2 - length/16) * 0.1
            retrieved_num = len(retrieved_set)
            #if retrieved_num < 10:
            #    reward += 0.1 * retrieved_num/10
            #else:
            #    reward += 0.1
            # 4) Boolean query format reward
            query = query.strip()
            query_upper = query.upper()
            has_and = ' AND ' in query_upper
            has_or = ' OR ' in query_upper
            has_boolean = has_and or has_or
            # Look for pattern like (text) AND/OR (text)
            phrase_pattern = r'\([^)]+\)\s+(AND|OR)\s+\([^)]+\)'
            has_phrase_pattern = bool(re.search(phrase_pattern, query_upper))
            if has_boolean:
                reward += 0.3 * 0.2
            if has_phrase_pattern:
                reward += 0.5 * 0.2
            if 5 < length < 30:
                reward += 0.2 * 0.2
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