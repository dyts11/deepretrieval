"""
GRPO Training Pipeline
Implements Group Relative Policy Optimization using TRL library
Based on stable configuration from experiment 8: LLaMA-3.2-3B + lr=1e-6
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mitigate CUDA memory fragmentation before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import time
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import wandb
from tqdm import tqdm
import warnings
from transformers.utils import logging as hf_logging

from utils.data_utils import load_deepretrieval_dataset
from models.reward_model import create_reward_model


def setup_grpo_config() -> GRPOConfig:
    """Setup GRPO configuration based on stable experiment 8 hyperparameters."""
    
    config = GRPOConfig(
        # Core training parameters (using correct GRPO parameter names)
        learning_rate=1e-6,                       # Proven stable learning rate
        #per_device_train_batch_size=4,            # Per-device batch size
        #gradient_accumulation_steps=4,            # Compensate for smaller batch size
        per_device_train_batch_size=1,            # Reduced for memory constraints
        gradient_accumulation_steps=8,
        num_train_epochs=1,                       # Based on 600 updates
        
        # GRPO-specific parameters
        #num_generations=4,                        # Generate 4 completions per prompt for comparison
        num_generations=2, 
        max_completion_length=32,                 # Same as max_new_tokens in PPO
        temperature=0.6,                          # Same generation parameters
        top_p=0.9,
        
        # KL control (adapted from PPO kl_coef=0.001)
        beta=0.001,                              # KL penalty coefficient
        
        # Optimization settings
        gradient_checkpointing=True,              # Memory optimization
        bf16=True,                               # Use bfloat16 for stability
            
        
        # Logging and output
        output_dir="models/grpo_training_output",
        logging_steps=1,
        
        # Wandb logging
        report_to="wandb",
        run_name="grpo-llama32-3b-experiment",
        
        # Other stable settings
        seed=42,
        dataloader_drop_last=True,
        remove_unused_columns=False,
    )
    
    return config


def setup_reward_function():
    """Setup reward function for GRPO using existing reward model."""
    print("🎯 Setting up reward model for GRPO...")
    
    # Use existing reward model with same configuration as PPO training
    reward_model = create_reward_model(top_k=100, reward_scale=1.0)
    
    def grpo_reward_function(completions: List[List[Dict[str, Any]]], **kwargs) -> List[float]:
        """
        GRPO reward function that processes multiple completions per prompt.
        Uses existing reward model's compute_rewards_batch method.
        
        Args:
            completions: List of completions, each completion is a list with dict containing 'content'
            **kwargs: Additional arguments (may include prompts, etc.)
            
        Returns:
            List of reward scores for each completion
        """
        # Extract queries from TRL completion format
        queries = []
        for completion in completions:
            try:
                if isinstance(completion, list) and len(completion) > 0:
                    content = completion[0].get("content", "")
                elif isinstance(completion, dict):
                    content = completion.get("content", "")
                else:
                    content = str(completion)
                queries.append(content)
            except Exception as e:
                print(f"Warning: Error extracting completion: {e}")
                queries.append("")  # Default empty query
        
        # For GRPO, we don't have relevant_doc_ids per completion
        # Use empty lists - the reward model will compute Boolean query rewards
        dummy_relevant = [[] for _ in queries]
        
        try:
            # Use existing batch reward computation
            rewards = reward_model.compute_rewards_batch(queries, dummy_relevant)
            return [float(r) for r in rewards]
        except Exception as e:
            print(f"Warning: Error computing batch rewards: {e}")
            return [0.0] * len(queries)
    
    # Quick smoke test
    test_completions = [[{"content": "stem cell AND pulmonary hypertension"}]]
    test_rewards = grpo_reward_function(test_completions)
    print(f"✅ Reward model ready - test reward: {test_rewards[0]:.3f}")
    
    return grpo_reward_function


def prepare_dataset_for_grpo() -> Dataset:
    """
    Prepare dataset for GRPO training using existing data loading.
    """
    print("📊 Loading dataset for GRPO training...")
    
    # Use existing data loading function
    raw_dataset = load_deepretrieval_dataset(
        data_path="data/train.jsonl", 
    )
    
    # Convert to format expected by GRPO
    # GRPO needs prompts as text, will generate completions automatically
    grpo_data = []
    for item in raw_dataset:
        grpo_data.append({
            "prompt": item["query"],  # The prompt for completion generation
            # Keep relevant_doc_ids for reference if needed later
            "relevant_doc_ids": item["relevant_doc_ids"]
        })
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(grpo_data)
    print(f"✅ Dataset prepared for GRPO: {len(dataset)} prompts")
    
    return dataset


def run_grpo_training() -> bool:
    """Run GRPO training with stable configuration."""
    print("🚀 GRPO Training Pipeline")
    print("=" * 50)
    
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Initialize wandb
    wandb.init(
        project="deepretrieval-grpo", 
        name="grpo-llama32-3b-experiment",
        config={
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "algorithm": "GRPO",
            "num_completions_per_prompt": 4,
            "learning_rate": 1e-6,
            "max_completion_length": 32,
        }
    )
    
    try:
        print("🤖 Setting up GRPO training...")
        
        # Setup components
        config = setup_grpo_config()
        reward_function = setup_reward_function()
        dataset = prepare_dataset_for_grpo()
        
        print("🔧 Initializing GRPO trainer...")
        
        # Create GRPO trainer
        trainer = GRPOTrainer(
            model="meta-llama/Llama-3.2-3B-Instruct",  # Stable model from experiment 8
            reward_funcs=reward_function,
            args=config,
            train_dataset=dataset,
            # TRL will handle tokenizer, reference model, etc. automatically
        )
        
        print("✅ GRPO trainer initialized")
        print("🚀 Starting GRPO training...")
        
        # Train the model (TRL handles the complex GRPO loop)
        trainer.train()
        
        wandb.finish()
        print("\n🎉 GRPO training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ GRPO training failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            wandb.finish()
        except Exception:
            pass
        return False


def main():
    """Main function"""
    success = run_grpo_training()
    return success


if __name__ == "__main__":
    main() 