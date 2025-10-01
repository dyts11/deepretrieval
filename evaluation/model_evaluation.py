"""
Model Evaluation for Trained GRPO/PPO Models
Evaluates retrieval performance using recall metrics on test dataset
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import pandas as pd

from utils.pubmed_api import PubmedAPI
from utils.data_utils import load_deepretrieval_dataset, format_pico_for_llm
from models.reward_model import create_reward_model, parse_boolean_query


class TrainedModelEvaluator:
    """
    Evaluates trained models (GRPO/PPO) using recall metrics on test dataset
    """
    
    def __init__(
        self,
        model_path: str,
        test_data_path: str = "data/test.jsonl",
        max_samples: int = 100,
        recall_k_values: List[int] = [50]
    ):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.max_samples = max_samples
        self.recall_k_values = recall_k_values
        
        self.model = None
        self.tokenizer = None
        self.pubmed_api = PubmedAPI()
        self.reward_model = create_reward_model(top_k=100)  # Match updated top_k value
        self.test_data = []
        
    def load_test_data(self) -> List[Dict]:
        """Load test dataset using existing data loading functions"""
        print(f"📊 Loading test data from {self.test_data_path}...")
        
        # Use existing data loading function
        dataset = load_deepretrieval_dataset(self.test_data_path, max_samples=self.max_samples)
        
        # Convert to our format
        test_data = []
        for item in dataset:
            test_data.append({
                'id': item['id'],
                'query_prompt': item['query'],  # Already formatted by format_pico_for_llm
                'relevant_pmids': item['relevant_doc_ids'],
            })
        
        print(f"✅ Loaded {len(test_data)} test samples")
        self.test_data = test_data
        return test_data
    

    
    def load_trained_model(self) -> bool:
        """Load the trained model from checkpoint"""
        print(f"🤖 Loading trained model from {self.model_path}...")
        
        try:
            # For GRPO/PPO models, they're usually saved as HuggingFace models
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ Model loaded successfully")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_queries(self, prompts: List[str]) -> List[str]:
        """Generate Boolean queries from prompts using trained model"""
        print(f"🔄 Generating queries for {len(prompts)} prompts...")
        
        generated_queries = []
        device = next(self.model.parameters()).device
        
        for i, prompt in enumerate(prompts):
            try:
                # Tokenize input
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate query
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=64,  # Shorter for Boolean queries
                        temperature=0.6,    # Same as training
                        top_p=0.9,         # Same as training
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode generated query
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                generated_queries.append(generated_text)
                
                if (i + 1) % 10 == 0:
                    print(f"   Generated {i + 1}/{len(prompts)} queries")
                    
            except Exception as e:
                print(f"❌ Error generating query for prompt {i}: {e}")
                generated_queries.append("")
        
        print(f"✅ Generated {len(generated_queries)} queries")
        return generated_queries
    
    def calculate_recall_at_k(
        self, 
        query: str, 
        relevant_pmids: List[str], 
        k: int
    ) -> float:
        """Calculate Recall@K for a single query using existing PubMed API"""
        try:
            # 1) Parse Boolean query into multiple search queries (same as reward model)
            search_queries = parse_boolean_query(query)
            
            # 2) Execute searches and combine results (same logic as reward model)
            all_retrieved_pmids = []
            pmid_scores = {}  # Track how many queries retrieved each PMID
            
            for search_query in search_queries:
                try:
                    pmids = self.pubmed_api.search_with_keywords(search_query, topk=k)
                    for pmid in pmids:
                        if pmid not in pmid_scores:
                            pmid_scores[pmid] = 0
                            all_retrieved_pmids.append(pmid)
                        pmid_scores[pmid] += 1
                except Exception as e:
                    print(f"Search failed for query '{search_query}': {e}")
                    continue
            
            # 3) Sort by relevance score (number of queries that retrieved this PMID)
            all_retrieved_pmids.sort(key=lambda pmid: pmid_scores[pmid], reverse=True)
            
            # 4) Take top-k results
            retrieved_pmids = all_retrieved_pmids[:k]
            
            # 5) Fallback: if no results from Boolean parsing, try original query
            if not retrieved_pmids:
                retrieved_pmids = self.pubmed_api.search_with_keywords(query, topk=k)
            
            # 6) Calculate recall using same logic as reward model
            relevant_set = set(relevant_pmids)
            retrieved_set = set(retrieved_pmids)  # Top-k results
            
            if len(relevant_set) == 0:
                return 0.0
            
            recall = len(retrieved_set & relevant_set) / len(relevant_set)
            return recall
            
        except Exception as e:
            print(f"❌ Error calculating recall for query '{query[:50]}...': {e}")
            return 0.0
    
    def evaluate_model_performance(self) -> Dict[str, Any]:
        """Evaluate model performance using recall metrics"""
        print("🎯 Evaluating model performance...")
        
        # Load test data
        if not self.test_data:
            self.load_test_data()
        
        # Generate queries
        prompts = [item['query_prompt'] for item in self.test_data]
        generated_queries = self.generate_queries(prompts)
        
        # Calculate recall for each K value
        recall_results = {f'recall@{k}': [] for k in self.recall_k_values}
        individual_results = []
        
        print(f"📊 Calculating recall metrics...")
        
        # Use batch processing for efficiency
        relevant_pmids_list = [item['relevant_pmids'] for item in self.test_data]
        
        for k in self.recall_k_values:
            print(f"   Calculating Recall@{k}...")
            
            for i, (test_item, query) in enumerate(zip(self.test_data, generated_queries)):
                if not query.strip():
                    # Empty query - zero recall
                    recall_results[f'recall@{k}'].append(0.0)
                else:
                    recall = self.calculate_recall_at_k(query, test_item['relevant_pmids'], k)
                    recall_results[f'recall@{k}'].append(recall)
                
                if (i + 1) % 10 == 0:
                    print(f"     Processed {i + 1}/{len(self.test_data)} samples")
        
        # Combine results for individual items
        for i, test_item in enumerate(self.test_data):
            item_recalls = {}
            for k in self.recall_k_values:
                item_recalls[f'recall@{k}'] = recall_results[f'recall@{k}'][i]
            
            individual_results.append({
                'id': test_item['id'],
                'prompt': test_item['query_prompt'],
                'generated_query': generated_queries[i],
                'relevant_pmids_count': len(test_item['relevant_pmids']),
                **item_recalls
            })
        
        # Calculate average recall scores
        avg_recall_scores = {}
        for k in self.recall_k_values:
            scores = recall_results[f'recall@{k}']
            avg_recall_scores[f'recall@{k}'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        results = {
            'model_path': self.model_path,
            'test_samples': len(self.test_data),
            'avg_recall_scores': avg_recall_scores,
            'individual_results': individual_results,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        print(f"\n✅ Evaluation completed!")
        print(f"📊 Results Summary:")
        for k in self.recall_k_values:
            mean_recall = avg_recall_scores[f'recall@{k}']['mean']
            print(f"   Recall@{k}: {mean_recall:.3f}")
        
        return results
    
    def compare_with_baseline(self, baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current model with baseline results"""
        print("🔍 Comparing with baseline...")
        
        current_results = self.evaluate_model_performance()
        
        comparison = {
            'current_model': self.model_path,
            'baseline_model': baseline_results.get('model_path', 'baseline'),
            'improvements': {}
        }
        
        for k in self.recall_k_values:
            current_recall = current_results['avg_recall_scores'][f'recall@{k}']['mean']
            baseline_recall = baseline_results['avg_recall_scores'][f'recall@{k}']['mean']
            
            improvement = current_recall - baseline_recall
            improvement_pct = (improvement / baseline_recall * 100) if baseline_recall > 0 else 0
            
            comparison['improvements'][f'recall@{k}'] = {
                'current': current_recall,
                'baseline': baseline_recall,
                'absolute_improvement': improvement,
                'relative_improvement_pct': improvement_pct
            }
        
        return comparison
    
    def save_results(self, results: Dict[str, Any], save_path: str):
        """Save evaluation results to file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {save_path}")
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate evaluation report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# Trained Model Evaluation Report

**Generated:** {timestamp}
**Model Path:** {results['model_path']}
**Test Samples:** {results['test_samples']}

## Recall Performance

"""
        
        for k in self.recall_k_values:
            stats = results['avg_recall_scores'][f'recall@{k}']
            report += f"""
### Recall@{k}
- **Mean:** {stats['mean']:.3f}
- **Std:** {stats['std']:.3f}
- **Median:** {stats['median']:.3f}
- **Min:** {stats['min']:.3f}
- **Max:** {stats['max']:.3f}
"""
        
        # Add sample results
        report += "\n## Sample Results\n"
        for i, item in enumerate(results['individual_results'][:5]):
            report += f"""
**Sample {i+1}:**
- **ID:** {item['id']}
- **Prompt:** {item['prompt'][:100]}...
- **Generated Query:** {item['generated_query']}
- **Relevant PMIDs:** {item['relevant_pmids_count']}
"""
            for k in self.recall_k_values:
                report += f"- **Recall@{k}:** {item[f'recall@{k}']:.3f}\n"
        
        return report


def evaluate_trained_model(
    model_path: str,
    test_data_path: str = "data/test.jsonl",
    max_samples: int = 100,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Main function to evaluate a trained model
    
    Args:
        model_path: Path to the trained model
        test_data_path: Path to test dataset
        max_samples: Number of test samples to use
        save_results: Whether to save results to file
    
    Returns:
        Evaluation results dictionary
    """
    evaluator = TrainedModelEvaluator(
        model_path=model_path,
        test_data_path=test_data_path,
        max_samples=max_samples
    )
    
    # Load model
    if not evaluator.load_trained_model():
        raise ValueError(f"Failed to load model from {model_path}")
    
    # Evaluate performance
    results = evaluator.evaluate_model_performance()
    
    # Save results
    #if save_results:
    #    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #    model_name = os.path.basename(model_path)
    #    save_path = f"evaluation/results/{model_name}_evaluation_{timestamp}.json"
    #    evaluator.save_results(results, save_path)
    #    
    #    # Generate and save report
    #    report = evaluator.generate_report(results)
    #    report_path = f"evaluation/results/{model_name}_report_{timestamp}.md"
    #    with open(report_path, 'w') as f:
    #        f.write(report)
    #    print(f"✅ Report saved to {report_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained model performance")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--test_data", default="data/test.jsonl", help="Path to test dataset")
    parser.add_argument("--max_samples", type=int, default=100, help="Max test samples")
    
    args = parser.parse_args()
    
    results = evaluate_trained_model(
        model_path=args.model_path,
        test_data_path=args.test_data,
        max_samples=args.max_samples
    )
    
    print("\n🎉 Evaluation completed!") 