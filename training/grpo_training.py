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
        per_device_train_batch_size=4,            # Reduced for memory constraints
        gradient_accumulation_steps=4,
        num_train_epochs=1,                       # Based on 600 updates
        
        # GRPO-specific parameters
        #num_generations=4,                        # Generate 4 completions per prompt for comparison
        num_generations=4, 
        max_completion_length=32,                 # Same as max_new_tokens in PPO
        temperature=0.6,                          # Same generation parameters
        top_p=0.9,
        
        # KL control (adapted from PPO kl_coef=0.001)
        beta=0.001,                              # KL penalty coefficient
        
        # Optimization settings
        gradient_checkpointing=True,              # Memory optimization
        bf16=True,                               # Use bfloat16 for stability
        dataloader_num_workers=0,                 # Reduce memory usage
        max_grad_norm=1.0,                       # Gradient clipping
             
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


def setup_reward_function(prompt_to_relevant_docs: Dict[str, List[str]]):
    """Setup reward function for GRPO using existing reward model."""
    print("🎯 Setting up reward model for GRPO...")
    
    # Use existing reward model with same configuration as PPO training
    reward_model = create_reward_model(top_k=100, reward_scale=1.0)
    
    def grpo_reward_function(prompts, completions, **kwargs) -> List[float]:
        """
        GRPO reward function that processes multiple completions per prompt.
        Uses existing reward model's compute_rewards_batch method.
        
        Args:
            prompts: List of original prompts used to generate completions
            completions: List of generated completions (strings)
            **kwargs: Additional arguments from dataset columns
            
        Returns:
            List of reward scores for each completion
        """
        # Get relevant documents for each completion
        # Map each prompt to its relevant documents
        relevant_docs_list = []
        
        for prompt in prompts:
            # Get relevant docs for this prompt
            relevant_docs = prompt_to_relevant_docs.get(prompt, [])
            relevant_docs_list.append(relevant_docs)
        
        # Debug: Check lengths match
        print(f"🔍 GRPO Debug - prompts: {len(prompts)}, completions: {len(completions)}, relevant_docs_list: {len(relevant_docs_list)}")
        if len(prompts) != len(completions) or len(completions) != len(relevant_docs_list):
            print(f"⚠️ Length mismatch! prompts={len(prompts)}, completions={len(completions)}, relevant_docs={len(relevant_docs_list)}")
        else:
            print(f"✅ Lengths match - all arrays have {len(completions)} items")
        
        try:
            # Use existing batch reward computation with relevant documents
            # completions are the generated queries to evaluate
            # relevant_docs_list contains the ground truth relevant docs for each prompt
            rewards = reward_model.compute_rewards_batch(completions, relevant_docs_list)
            print(f"🔍 GRPO Debug - computed {len(rewards)} rewards, first few: {rewards[:3]}")
            return [float(r) for r in rewards]
        except Exception as e:
            print(f"❌ Error computing batch rewards: {e}")
            print(f"Debug info - prompts len: {len(prompts)}, completions len: {len(completions)}, relevant_docs len: {len(relevant_docs_list)}")
            return [0.0] * len(completions)
    
    
    return grpo_reward_function


def prepare_dataset_for_grpo() -> tuple[Dataset, Dict[str, List[str]]]:
    """
    Prepare dataset for GRPO training using existing data loading.
    
    Returns:
        Dataset: GRPO-formatted dataset
        Dict: Mapping from prompts to their relevant document IDs
    """
    print("📊 Loading dataset for GRPO training...")
    
    # Use existing data loading function
    raw_dataset = load_deepretrieval_dataset(
        data_path="data/full_train.jsonl",
        max_samples=100  # Use 1000 samples for meaningful training
    )
    
    # Create mapping from prompts to relevant documents
    prompt_to_relevant_docs = {}
    
    # Convert to format expected by GRPO
    # GRPO needs prompts as text, will generate completions automatically
    grpo_data = []
    for item in raw_dataset:
        prompt = item["query"]
        relevant_docs = item["relevant_doc_ids"]
        
        grpo_data.append({
            "prompt": prompt,  # The prompt for completion generation
        })
        
        # Store mapping for reward computation
        prompt_to_relevant_docs[prompt] = relevant_docs
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(grpo_data)
    print(f"✅ Dataset prepared for GRPO: {len(dataset)} prompts")
    print(f"✅ Created prompt-to-relevant-docs mapping with {len(prompt_to_relevant_docs)} entries")
    
    return dataset, prompt_to_relevant_docs


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
        #name="grpo-qwen2-0.5b-experiment",
        config={
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            #"model": "Qwen/Qwen2-0.5B-Instruct",
            "algorithm": "GRPO",
            "num_generations": 2,
            "learning_rate": 1e-6,
            "max_completion_length": 32,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 8,
        }
    )
    
    try:
        print("🤖 Setting up GRPO training...")
        
        # Setup components
        config = setup_grpo_config()
        dataset, prompt_to_relevant_docs = prepare_dataset_for_grpo()
        reward_function = setup_reward_function(prompt_to_relevant_docs)
        
        print("🔧 Initializing GRPO trainer...")
        
        # Create GRPO trainer
        trainer = GRPOTrainer(
            "meta-llama/Llama-3.2-3B-Instruct",  # Stable model from experiment 8
            #model="Qwen/Qwen2-0.5B-Instruct",
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