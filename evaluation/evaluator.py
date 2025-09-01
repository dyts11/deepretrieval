"""
Phase 6: Evaluation Module
Evaluates model performance, compares different models, and generates reports
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, T5ForConditionalGeneration
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from utils.data_utils import load_deepretrieval_dataset
from models.reward_model import create_reward_model


class ModelEvaluator:
    """
    Evaluates model performance for query augmentation and retrieval
    """
    
    def __init__(
        self,
        model_path: str,
        reward_model=None,
        test_dataset=None,
        max_samples: int = 100
    ):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.reward_model = reward_model or create_reward_model()
        self.test_dataset = test_dataset
        self.max_samples = max_samples
        self.evaluation_results = {}
        
    def load_model(self):
        """Load the trained model"""
        print(f"🤖 Loading model from {self.model_path}...")
        
        try:
            # Check if model path exists and is valid
            if self.model_path is None or not os.path.exists(self.model_path):
                print(f"❌ Model path does not exist: {self.model_path}")
                print("   Using base model for testing...")
                # Fallback to base model for testing
                self.model = T5ForConditionalGeneration.from_pretrained(
                    "google/flan-t5-small",
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            else:
                # Check if tokenizer files exist
                tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
                has_tokenizer = any(os.path.exists(os.path.join(self.model_path, f)) for f in tokenizer_files)
                
                if has_tokenizer:
                    # Load both model and tokenizer from the path
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        self.model_path,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                else:
                    # Load model from path, tokenizer from base model
                    print(f"⚠️ Model exists but tokenizer files missing, loading tokenizer from base model")
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        self.model_path,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            
            # T5 tokenizer doesn't need pad_token = eos_token
            
            print(f"✅ Model loaded successfully")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   Device: {next(self.model.parameters()).device}")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def prepare_test_dataset(self):
        """Prepare test dataset for evaluation"""
        if self.test_dataset is None:
            print("📊 Loading test dataset...")
            self.test_dataset = load_deepretrieval_dataset(
                data_path="data/train.jsonl",
                max_samples=self.max_samples
            )
            print(f"✅ Test dataset loaded: {len(self.test_dataset)} samples")
        
        return self.test_dataset
    
    def generate_queries(self, prompts: List[str]) -> List[str]:
        """Generate queries from prompts"""
        queries = []
        
        # Get device from model
        device = next(self.model.parameters()).device
        print(f"Using device: {device}")
        
        for prompt in prompts:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                query = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                queries.append(query)
                
            except Exception as e:
                print(f"❌ Error generating query: {e}")
                queries.append("")
        
        return queries
    
    def evaluate_retrieval_performance(self) -> Dict[str, Any]:
        """Evaluate retrieval performance using reward model"""
        print("🎯 Evaluating retrieval performance...")
        
        dataset = self.prepare_test_dataset()
        prompts = [item["query"] for item in dataset]
        relevant_pmids_list = [item["relevant_doc_ids"] for item in dataset]
        
        # Generate queries
        generated_queries = self.generate_queries(prompts)
        
        # Compute rewards
        rewards = []
        for i, query in enumerate(generated_queries):
            relevant_pmids = relevant_pmids_list[i]
            reward = self.reward_model.compute_reward(query, relevant_pmids)
            rewards.append(reward)
        
        # Calculate statistics
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        max_reward = np.max(rewards)
        min_reward = np.min(rewards)
        
        # Reward distribution
        reward_distribution = {
            'excellent': len([r for r in rewards if r >= 0.8]),
            'good': len([r for r in rewards if 0.5 <= r < 0.8]),
            'fair': len([r for r in rewards if 0.2 <= r < 0.5]),
            'poor': len([r for r in rewards if r < 0.2])
        }
        
        results = {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'total_queries': len(rewards),
            'reward_distribution': reward_distribution,
            'individual_rewards': rewards,
            'generated_queries': generated_queries,
            'prompts': prompts
        }
        
        print(f"✅ Retrieval evaluation completed")
        print(f"   Average reward: {avg_reward:.3f}")
        print(f"   Reward std: {std_reward:.3f}")
        print(f"   Max reward: {max_reward:.3f}")
        print(f"   Min reward: {min_reward:.3f}")
        
        return results
    
    def evaluate_query_quality(self) -> Dict[str, Any]:
        """Evaluate query quality metrics"""
        print("📝 Evaluating query quality...")
        
        dataset = self.prepare_test_dataset()
        prompts = [item["query"] for item in dataset]
        generated_queries = self.generate_queries(prompts)
        
        # Quality metrics
        query_lengths = [len(query.split()) for query in generated_queries]
        avg_length = np.mean(query_lengths)
        
        # Check for repetition
        repetitive_queries = 0
        for query in generated_queries:
            words = query.split()
            if len(words) > 3:
                unique_words = len(set(words))
                if unique_words / len(words) < 0.5:  # More than 50% repetition
                    repetitive_queries += 1
        
        # Check for empty or very short queries
        empty_queries = len([q for q in generated_queries if len(q.strip()) < 5])
        
        quality_results = {
            'avg_query_length': avg_length,
            'repetitive_queries': repetitive_queries,
            'empty_queries': empty_queries,
            'query_lengths': query_lengths,
            'generated_queries': generated_queries
        }
        
        print(f"✅ Query quality evaluation completed")
        print(f"   Average query length: {avg_length:.1f} words")
        print(f"   Repetitive queries: {repetitive_queries}")
        print(f"   Empty queries: {empty_queries}")
        
        return quality_results
    
    def compare_models(self, model_paths: List[str]) -> Dict[str, Any]:
        """Compare multiple models"""
        print("🔍 Comparing models...")
        
        comparison_results = {}
        
        for model_path in model_paths:
            print(f"\n📊 Evaluating {model_path}...")
            
            # Create evaluator for this model
            evaluator = ModelEvaluator(model_path, self.reward_model, self.test_dataset)
            if evaluator.load_model():
                retrieval_results = evaluator.evaluate_retrieval_performance()
                quality_results = evaluator.evaluate_query_quality()
                
                comparison_results[model_path] = {
                    'retrieval': retrieval_results,
                    'quality': quality_results
                }
        
        return comparison_results
    
    def generate_evaluation_report(self, save_path: str = None) -> str:
        """Generate comprehensive evaluation report"""
        print("📋 Generating evaluation report...")
        
        # Run evaluations
        retrieval_results = self.evaluate_retrieval_performance()
        quality_results = self.evaluate_query_quality()
        
        # Generate report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# Model Evaluation Report
**Generated:** {timestamp}
**Model Path:** {self.model_path}
**Dataset Size:** {len(self.test_dataset)} samples

## Retrieval Performance
- **Average Reward:** {retrieval_results['avg_reward']:.3f}
- **Reward Std:** {retrieval_results['std_reward']:.3f}
- **Max Reward:** {retrieval_results['max_reward']:.3f}
- **Min Reward:** {retrieval_results['min_reward']:.3f}

## Reward Distribution
- **Excellent (≥0.8):** {retrieval_results['reward_distribution']['excellent']}
- **Good (0.5-0.8):** {retrieval_results['reward_distribution']['good']}
- **Fair (0.2-0.5):** {retrieval_results['reward_distribution']['fair']}
- **Poor (<0.2):** {retrieval_results['reward_distribution']['poor']}

## Query Quality
- **Average Length:** {quality_results['avg_query_length']:.1f} words
- **Repetitive Queries:** {quality_results['repetitive_queries']}
- **Empty Queries:** {quality_results['empty_queries']}

## Sample Generated Queries
"""
        
        # Add sample queries
        for i, (prompt, query) in enumerate(zip(quality_results['generated_queries'][:5], retrieval_results['prompts'][:5])):
            report += f"""
**Sample {i+1}:**
- **Prompt:** {prompt[:100]}...
- **Generated Query:** {query[:100]}...
- **Reward:** {retrieval_results['individual_rewards'][i]:.3f}
"""
        
        # Save report
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"✅ Report saved to {save_path}")
        
        return report
    
    def plot_results(self, save_path: str = None):
        """Create visualization plots"""
        print("📊 Creating visualization plots...")
        
        retrieval_results = self.evaluate_retrieval_performance()
        quality_results = self.evaluate_query_quality()
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward distribution
        rewards = retrieval_results['individual_rewards']
        axes[0, 0].hist(rewards, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        
        # Query length distribution
        lengths = quality_results['query_lengths']
        axes[0, 1].hist(lengths, bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Query Length Distribution')
        axes[0, 1].set_xlabel('Query Length (words)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Reward vs Query Length
        axes[1, 0].scatter(lengths, rewards, alpha=0.6, color='orange')
        axes[1, 0].set_title('Reward vs Query Length')
        axes[1, 0].set_xlabel('Query Length (words)')
        axes[1, 0].set_ylabel('Reward')
        
        # Reward categories
        categories = list(retrieval_results['reward_distribution'].keys())
        counts = list(retrieval_results['reward_distribution'].values())
        axes[1, 1].bar(categories, counts, color=['red', 'orange', 'yellow', 'green'])
        axes[1, 1].set_title('Reward Categories')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plots saved to {save_path}")
        
        plt.show()


def create_evaluator(model_path: str, max_samples: int = 100) -> ModelEvaluator:
    """Factory function to create model evaluator"""
    return ModelEvaluator(model_path, max_samples=max_samples) 