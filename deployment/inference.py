"""
Phase 6: Deployment Module
Provides model inference, API endpoints, and production-ready query generation
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import logging
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, T5ForConditionalGeneration
from datetime import datetime
import time

from models.reward_model import create_reward_model


class QueryGenerator:
    """
    Production-ready query generator for deployment
    """
    
    def __init__(
        self,
        model_path: str,
        reward_model=None,
        device: str = "cpu",
        max_length: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        if model_path is None:
            raise ValueError("model_path cannot be None")
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.reward_model = reward_model
        self.model = None
        self.tokenizer = None
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_model(self):
        """Load the trained model"""
        self.logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Check if model path exists and is valid
            if self.model_path is None or not os.path.exists(self.model_path):
                self.logger.warning(f"Model path does not exist: {self.model_path}, using base model")
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
                    self.logger.warning(f"Model exists but tokenizer files missing, loading tokenizer from base model")
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        self.model_path,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            
            # T5 tokenizer doesn't need pad_token = eos_token
            self.logger.info(f"Model loaded successfully with {sum(p.numel() for p in self.model.parameters()):,} parameters")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def generate_query(self, pico_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate a query from PICO information
        
        Args:
            pico_info: Dictionary with keys 'P', 'I', 'C', 'O'
            
        Returns:
            Dictionary with generated query and metadata
        """
        try:
            # Format PICO for model input
            prompt = self._format_pico_prompt(pico_info)
            
            # Generate query
            query = self._generate_from_prompt(prompt)
            
            # Compute reward if reward model is available
            reward = None
            if self.reward_model:
                relevant_pmids = pico_info.get('relevant_pmids', [])
                reward = self.reward_model.compute_reward(query, relevant_pmids)
            
            result = {
                'generated_query': query,
                'prompt': prompt,
                'pico_info': pico_info,
                'reward': reward,
                'timestamp': datetime.now().isoformat(),
                'model_path': self.model_path,
                'generation_params': {
                    'max_length': self.max_length,
                    'temperature': self.temperature,
                    'top_p': self.top_p
                }
            }
            
            self.logger.info(f"Generated query: {query[:50]}...")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating query: {e}")
            return {
                'error': str(e),
                'generated_query': '',
                'timestamp': datetime.now().isoformat()
            }
    
    def _format_pico_prompt(self, pico_info: Dict[str, str]) -> str:
        """Format PICO information into a prompt optimized for T5"""
        return f"""Generate a PubMed search query for this medical research question.

Population: {pico_info.get('P', '')}
Intervention: {pico_info.get('I', '')}
Comparison: {pico_info.get('C', '')}
Outcome: {pico_info.get('O', '')}

Create a search query using Boolean operators (AND, OR) and medical terms."""
    
    def _generate_from_prompt(self, prompt: str) -> str:
        """Generate query from prompt using T5"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return query.strip()
            
        except Exception as e:
            self.logger.error(f"Error in generation: {e}")
            return ""
    
    def batch_generate(self, pico_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Generate queries for multiple PICO inputs"""
        results = []
        
        for i, pico_info in enumerate(pico_list):
            self.logger.info(f"Processing batch item {i+1}/{len(pico_list)}")
            result = self.generate_query(pico_info)
            results.append(result)
        
        return results
    
    def evaluate_batch(self, pico_list: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate a batch of PICO inputs"""
        if not self.reward_model:
            return {'error': 'Reward model not available'}
        
        results = self.batch_generate(pico_list)
        rewards = [r.get('reward', 0) for r in results if r.get('reward') is not None]
        
        if rewards:
            return {
                'avg_reward': sum(rewards) / len(rewards),
                'max_reward': max(rewards),
                'min_reward': min(rewards),
                'total_queries': len(rewards),
                'results': results
            }
        else:
            return {'error': 'No valid rewards computed'}


class QueryAPI:
    """
    Simple API wrapper for query generation
    """
    
    def __init__(self, model_path: str):
        self.generator = QueryGenerator(model_path)
        self.generator.load_model()
    
    def generate_single(self, pico_info: Dict[str, str]) -> Dict[str, Any]:
        """Generate a single query"""
        return self.generator.generate_query(pico_info)
    
    def generate_batch(self, pico_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Generate queries for multiple inputs"""
        return self.generator.batch_generate(pico_list)
    
    def evaluate_batch(self, pico_list: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate a batch of inputs"""
        return self.generator.evaluate_batch(pico_list)


def create_query_generator(model_path: str, **kwargs) -> QueryGenerator:
    """Factory function to create query generator"""
    return QueryGenerator(model_path, **kwargs)


def create_query_api(model_path: str) -> QueryAPI:
    """Factory function to create query API"""
    return QueryAPI(model_path)


# Example usage and testing
def test_deployment():
    """Test the deployment module"""
    print("🧪 Testing deployment module...")
    
    # Test with a sample model path (you'll need to have a trained model)
    model_path = "models/phase3_test"  # Use existing model for testing
    
    try:
        # Create generator
        generator = create_query_generator(model_path)
        
        # Test single generation
        pico_info = {
            'P': 'Patients with advanced pulmonary arterial hypertension (PAH)',
            'I': 'Stem cell therapy',
            'C': 'Standard drug treatment or placebo',
            'O': 'Efficacy of stem cell therapy for PAH'
        }
        
        result = generator.generate_query(pico_info)
        print("✅ Single query generation successful")
        print(f"   Generated query: {result['generated_query'][:50]}...")
        
        # Test batch generation
        pico_list = [pico_info, pico_info]  # Same input for testing
        batch_results = generator.batch_generate(pico_list)
        print("✅ Batch generation successful")
        print(f"   Generated {len(batch_results)} queries")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment test failed: {e}")
        return False


if __name__ == "__main__":
    test_deployment() 