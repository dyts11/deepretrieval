import json
import requests
import traceback
import time
import pandas as pd
import xml.etree.ElementTree as ET
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
import os
import re
from collections import OrderedDict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PubMed API configuration
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term="
PUBMED_SUMMARY_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id="
PUBMED_EFETCH_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id="


class LocalCorpusRetriever:
    """Simple offline retriever using TF-IDF over a local corpus.
    Expects a JSON file with a list of {"doc_id": str, "text": str} entries.
    """
    def __init__(self, corpus_path: Optional[str] = None):
        self.doc_ids: List[str] = []
        self.texts: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_matrix = None
        if corpus_path and os.path.exists(corpus_path):
            try:
                with open(corpus_path, "r") as f:
                    data = json.load(f)
                self.doc_ids = [d.get("doc_id", str(i)) for i, d in enumerate(data)]
                self.texts = [d.get("text", "") for d in data]
                self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
                self.doc_matrix = self.vectorizer.fit_transform(self.texts)
            except Exception as e:
                print(f"⚠️ Failed to build offline corpus from {corpus_path}: {e}")
                self.doc_ids = []
                self.texts = []
                self.vectorizer = None
                self.doc_matrix = None

    def is_ready(self) -> bool:
        return self.vectorizer is not None and self.doc_matrix is not None and len(self.doc_ids) > 0

    def search(self, query: str, topk: int = 10) -> List[str]:
        if not self.is_ready():
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        if topk <= 0:
            topk = len(self.doc_ids)
        top_idx = np.argsort(-sims)[:topk]
        return [self.doc_ids[i] for i in top_idx]


class PubmedAPI:
    """A wrapper class for the Pubmed API with optimized performance and offline/caching support."""

    def __init__(
        self,
        retry: int = 3,
        api_key: Optional[str] = None,
        request_delay: float = 0.1,
        offline: bool = False,
        cache_path: Optional[str] = None,
        offline_corpus_path: Optional[str] = None,
        cache_max_size: int = 50000,
    ):
        self.retry = retry
        # API key from arg or environment variable
        self.api_key = api_key if api_key is not None else os.environ.get("PUBMED_API_KEY")
        self.request_delay = request_delay  # Rate limiting: 0.1s = 10 rps
        self.last_request_time = 0.0
        self.semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

        # Offline support
        self.offline = offline
        self.offline_retriever = LocalCorpusRetriever(offline_corpus_path) if offline else None
        if self.offline and self.offline_retriever and not self.offline_retriever.is_ready():
            print("⚠️ Offline mode requested but corpus is not ready; falling back to empty results unless cache has entries.")

        # Caching (in-memory LRU + optional disk persistence)
        self.cache: OrderedDict[str, List[str]] = OrderedDict()
        self.cache_path = cache_path
        self.cache_max_size = cache_max_size
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        self.cache[entry["key"]] = entry["pmids"]
                # keep LRU size bound
                while len(self.cache) > self.cache_max_size:
                    self.cache.popitem(last=False)
                print(f"✅ Loaded {len(self.cache)} cached queries from {self.cache_path}")
            except Exception as e:
                print(f"⚠️ Failed to load cache from {self.cache_path}: {e}")

    # -----------------------------
    # Caching helpers
    # -----------------------------
    def _cache_key(self, query: str, topk: int) -> str:
        return f"{query}||k={topk}"

    def _cache_get(self, query: str, topk: int) -> Optional[List[str]]:
        key = self._cache_key(query, topk)
        if key in self.cache:
            pmids = self.cache.pop(key)
            self.cache[key] = pmids  # move to end (LRU)
            return pmids
        return None

    def _cache_set(self, query: str, topk: int, pmids: List[str]):
        key = self._cache_key(query, topk)
        if key in self.cache:
            self.cache.pop(key)
        self.cache[key] = pmids
        if len(self.cache) > self.cache_max_size:
            self.cache.popitem(last=False)
        # Persist incrementally if a path is provided (JSONL append)
        if self.cache_path:
            try:
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"key": key, "pmids": pmids}) + "\n")
            except Exception:
                pass

    # -----------------------------
    # Async methods
    # -----------------------------
    async def search_with_query_async(self, query, topk=-1):
        """Async search handling rate limiting, caching, and offline mode."""
        # Offline path
        if self.offline and self.offline_retriever:
            cached = self._cache_get(query, topk)
            if cached is not None:
                return cached
            pmids = self.offline_retriever.search(query, topk=topk if topk > 0 else 10)
            self._cache_set(query, topk, pmids)
            return pmids

        # Cache check for online path
        cached = self._cache_get(query, topk)
        if cached is not None:
            return cached

        async with self.semaphore:
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.request_delay:
                await asyncio.sleep(self.request_delay - time_since_last)
            
            # Ensure proper URL format for PubMed API
            if not query.startswith("http"):
                query = PUBMED_BASE_URL + query
            
            # Add API key if available
            if self.api_key:
                query += f"&api_key={self.api_key}"
            
            # JSON response
            if "retmode=json" not in query:
                query += "&retmode=json"
            
            # Prepare retmax
            retstart = 0
            retmax = topk if (topk > 0 and topk <= 1000) else 1000
            
            for attempt in range(self.retry + 1):
                try:
                    count_query = query + f"&retstart={retstart}&retmax={retmax}"
                    async with aiohttp.ClientSession() as session:
                        async with session.get(count_query, timeout=30) as response:
                            if response.status == 200:
                                response_text = await response.text()
                                response_dict = json.loads(response_text)
                                if "esearchresult" in response_dict:
                                    id_list = response_dict["esearchresult"].get("idlist", [])
                                    self.last_request_time = time.time()
                                    # Cache and return
                                    self._cache_set(query, topk, id_list[:topk] if topk > 0 else id_list)
                                    return id_list[:topk] if topk > 0 else id_list
                                else:
                                    return []
                            else:
                                if attempt < self.retry:
                                    await asyncio.sleep(1)
                                    continue
                                else:
                                    raise ConnectionError(f"PubMed HTTP {response.status}")
                except Exception:
                    if attempt < self.retry:
                        await asyncio.sleep(2)
                        continue
                    else:
                        return []

    async def search_with_keywords_async(self, keywords: str, topk=-1):
        query = PUBMED_BASE_URL + keywords if not self.offline else keywords
        return await self.search_with_query_async(query, topk=topk)

    async def compute_reward_async(self, query: str, relevant_pmids: list) -> float:
        if not relevant_pmids:
            return 0.0
        retrieved_pmids = await self.search_with_keywords_async(query, topk=10)
        relevant_set = set(relevant_pmids)
        retrieved_set = set(retrieved_pmids)
        recall = len(retrieved_set & relevant_set) / len(relevant_set)
        return recall

    # -----------------------------
    # Sync methods
    # -----------------------------
    def search_with_keywords(self, keywords: str, topk=-1):
        """Search with keywords input to get a ranked list of pmids."""
        # Offline path
        if self.offline and self.offline_retriever:
            cached = self._cache_get(keywords, topk)
            if cached is not None:
                return cached
            pmids = self.offline_retriever.search(keywords, topk=topk if topk > 0 else 10)
            self._cache_set(keywords, topk, pmids)
            return pmids

        query = PUBMED_BASE_URL + keywords
        return self.search_with_query(query, topk=topk)
    
    def search_with_keywords_batch(self, keywords_list: list, topk=-1):
        """Search multiple keywords in parallel using threading"""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        def search_single(keyword):
            try:
                pmids = self.search_with_keywords(keyword, topk)
                return keyword, pmids
            except Exception as e:
                return keyword, []
        
        max_workers = min(5, len(keywords_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_keyword = {executor.submit(search_single, keyword): keyword for keyword in keywords_list}
            for future in as_completed(future_to_keyword):
                keyword, pmids = future.result()
                results[keyword] = pmids
        
        return results

    def search_with_query(self, query, topk=-1):
        err_msg = ""

        # Cache check (use raw query as key)
        cached = self._cache_get(query, topk)
        if cached is not None:
            return cached

        # Ensure proper URL format for PubMed API
        if '?' in query:
            separator = '&'
        else:
            separator = '?'
            
        if 'retmode=json' not in query:
            query += f"{separator}retmode=json"
            
        if self.api_key:
            query += f"&api_key={self.api_key}"
        
        retstart = 0
        if topk > 0 and topk <= 1000:
            retmax = topk
        else:
            retmax = 5000
        original_query = query
        pmid_list = []
        for i in range(self.retry + 1):
            if i > 0:
                time.sleep(self.request_delay * 2)
            try:
                count_query = original_query + f'&retmax=0'
                response = requests.get(count_query, timeout=30)
                if response.status_code != 200:
                    raise ConnectionError(f"Pubmed connection error occurred - {response.text}")
                
                if not response.text.strip():
                    raise ValueError("Empty response from PubMed API")
                
                try:
                    response_dict = json.loads(response.text)
                except json.JSONDecodeError as json_err:
                    raise ValueError(f"Invalid JSON response: {json_err}")
                
                if 'esearchresult' not in response_dict:
                    raise ValueError("Response missing 'esearchresult' key")
                if 'count' not in response_dict['esearchresult']:
                    raise ValueError("Response missing 'count' key")
                
                total_count = int(response_dict['esearchresult']['count'])
                count_to_fetch = min(total_count, topk) if topk > 0 else total_count
                if count_to_fetch <= 0:
                    return []
                batch_limit = min(10000, count_to_fetch)
                while retstart < batch_limit and len(pmid_list) < count_to_fetch:
                    current_retmax = min(retmax, count_to_fetch - len(pmid_list))
                    current_query = original_query + f'&retmax={current_retmax}&retstart={retstart}'
                    response = requests.get(current_query, timeout=30)
                    time.sleep(self.request_delay)
                    if response.status_code != 200:
                        raise ConnectionError(f"Pubmed connection error occurred - {response.text}")
                    
                    if not response.text.strip():
                        raise ValueError("Empty response from PubMed API")
                    
                    try:
                        response_dict = json.loads(response.text)
                    except json.JSONDecodeError as json_err:
                        raise ValueError(f"Invalid JSON response: {json_err}")
                    
                    if 'esearchresult' not in response_dict:
                        raise ValueError("Response missing 'esearchresult' key")
                    if 'idlist' not in response_dict['esearchresult']:
                        raise ValueError("Response missing 'idlist' key")
                    
                    batch_results = response_dict['esearchresult']['idlist']
                    if not batch_results:
                        break
                    pmid_list.extend(batch_results)
                    retstart += current_retmax
                    if topk > 0 and len(pmid_list) >= topk:
                        pmid_list = pmid_list[:topk]
                        break
                if topk <= 0:
                    pmid_list = list(dict.fromkeys(pmid_list))
                break
            except Exception as e:
                err_msg = traceback.format_exc()
        if err_msg != "":
            return []

        # Cache the result
        self._cache_set(query, topk, pmid_list)
        return pmid_list

    def get_papers_by_pmids(self, pmid_list):
        """Search pmids to get the summary of paper."""
        if not pmid_list:
            return pd.DataFrame()
        err_msg = ""
        for i in range(self.retry + 1):
            if i > 0:
                time.sleep(self.request_delay * 2)
            try:
                batch_size = 200
                all_papers = []
                for i in range(0, len(pmid_list), batch_size):
                    batch_pmids = pmid_list[i:i+batch_size]
                    pmid_list_str = ','.join(batch_pmids)
                    summary_query = PUBMED_SUMMARY_BASE_URL + pmid_list_str + "&retmode=json"
                    if self.api_key:
                        summary_query += f"&api_key={self.api_key}"
                    response = requests.get(summary_query)
                    time.sleep(self.request_delay)
                    if response.status_code != 200:
                        if response.status_code == 414:
                            raise ConnectionError(f"Pubmed query too long!")
                        else:
                            raise ConnectionError(f"Pubmed connection error occurred: {response.text}")
                    response = json.loads(response.text)
                    results = response.get("result", {})
                    uids = results.get("uids", [])
                    if len(uids) == 0:
                        continue
                    batch_papers = []
                    for idx in range(len(uids)):
                        try:
                            cur_res = results[uids[idx]]
                            parse_res = self._parse_json_summary_response(cur_res)
                            batch_papers.append(parse_res)
                        except Exception:
                            pass
                    if batch_papers:
                        batch_df = pd.concat(batch_papers, axis=0).reset_index(drop=True)
                        abstracts, mesh_terms = self._retrieve_abstract_and_mesh_term_by_efetch(uids)
                        batch_df['Abstract'] = abstracts
                        batch_df['Mesh Term'] = mesh_terms
                        all_papers.append(batch_df)
                if all_papers:
                    papers = pd.concat(all_papers, axis=0).reset_index(drop=True)
                else:
                    papers = pd.DataFrame()
                break
            except Exception:
                err_msg = traceback.format_exc()
        if err_msg != "":
            raise RuntimeError("A Pubmed API error occurred")
        return papers

    def compute_reward(self, query: str, relevant_pmids: list) -> float:
        """
        Compute reward based on recall@k for TRL training (legacy helper, kept for compatibility)
        """
        if not relevant_pmids:
            return 0.0
        retrieved_pmids = self.search_with_keywords(query, topk=10)
        relevant_set = set(relevant_pmids)
        retrieved_set = set(retrieved_pmids)
        recall = len(retrieved_set & relevant_set) / len(relevant_set)
        return recall

    def test_api_connection(self) -> bool:
        """Test if the API connection works"""
        try:
            test_pmids = self.search_with_keywords("diabetes", topk=2)
            return bool(test_pmids)
        except Exception:
            return False

    def _parse_json_summary_response(self, res):
        pmid = res.get("uid", None)
        pub_date = res.get("pubdate", None)
        journal = res.get("fulljournalname", None)
        if journal is None:
            journal = res.get("source", None)
        title = res.get("title", None)
        authors = res.get("authors", [])
        authors = [author.get("name", None) for author in authors]
        authors = "; ".join(authors)
        volume = res.get("volume", None)
        issue = res.get("issue", None)
        pages = res.get("pages", None)
        pubtypes = res.get("pubtype", [])
        url = 'https://pubmed.ncbi.nlm.nih.gov/' + pmid
        df = pd.DataFrame({
            "PMID": [pmid],
            "Publication Date": [pub_date],
            "Title": [title],
            "Authors": [authors],
            "Journal": [journal],
            "Volume": [volume],
            "Issue": [issue],
            "Pages": [pages],
            "Pubtypes": [pubtypes],
            "URL": [url]
        })
        return df

    def _retrieve_abstract_and_mesh_term_by_efetch(self, pmid_list):
        batch_size = 200
        all_abstracts = {}
        all_mesh_terms = {}
        for i in range(0, len(pmid_list), batch_size):
            batch_pmids = pmid_list[i:i+batch_size]
            pmid_list_str = ','.join(batch_pmids)
            query = PUBMED_EFETCH_BASE_URL + pmid_list_str + "&retmode=xml"
            if self.api_key:
                query += "&api_key=" + self.api_key
            response = requests.get(query)
            time.sleep(self.request_delay)
            if response.status_code != 200:
                continue
            try:
                response_text = response.text
                tree = ET.ElementTree(ET.fromstring(response_text))
                articles = tree.findall(".//PubmedArticle")
                for article in articles:
                    abstract = article.find(".//AbstractText")
                    if abstract is not None:
                        abstract_text = abstract.text
                    else:
                        abstract_text = ""
                    article_ids = [a for a in article.findall(".//ArticleId") if a.get("IdType").lower() == "pubmed"]
                    if len(article_ids) > 0:
                        pmid = article_ids[0].text
                        all_abstracts[pmid] = abstract_text
                    else:
                        continue
                    mesh_terms = []
                    mesh_headings = article.findall(".//MeshHeading")
                    for mesh_heading in mesh_headings:
                        descriptor = mesh_heading.find("DescriptorName")
                        if descriptor is not None:
                            qualifiers = mesh_heading.findall("QualifierName")
                            if qualifiers:
                                for qualifier in qualifiers:
                                    mesh_terms.append(f"{descriptor.text} ({qualifier.text})")
                            else:
                                mesh_terms.append(descriptor.text)
                    all_mesh_terms[pmid] = mesh_terms
            except Exception:
                pass
        output_abstracts = []
        output_mesh_terms = []
        for pmid in pmid_list:
            output_abstracts.append(all_abstracts.get(pmid, ""))
            output_mesh_terms.append(all_mesh_terms.get(pmid, []))
        return output_abstracts, output_mesh_terms


# Factory function for TRL integration

def create_pubmed_retriever(api_key: Optional[str] = None, offline: bool = False, cache_path: Optional[str] = None, offline_corpus_path: Optional[str] = None):
    """Create a PubMed retriever for TRL training.
    If `offline` is True, or if no API key is available, the retriever will use a local TF-IDF index when possible.
    """
    if api_key is None:
        api_key = os.environ.get("PUBMED_API_KEY")
    use_offline = offline
    return PubmedAPI(api_key=api_key, offline=use_offline, cache_path=cache_path, offline_corpus_path=offline_corpus_path) 