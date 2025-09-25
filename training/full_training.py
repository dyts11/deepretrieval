"""
Phase 5: Full Training Pipeline
Implements actual PPO training with PPO updates, random mini-batches, and cleaned config
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mitigate CUDA memory fragmentation before importing torch
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import time
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import wandb
from tqdm import tqdm
import inspect
import warnings
import re
from transformers.utils import logging as hf_logging

from utils.data_utils import load_deepretrieval_dataset
from models.reward_model import create_reward_model


def setup_ppo_config(device: str) -> PPOConfig:
    """Setup PPO configuration and auto-filter unsupported keys for TRL version compatibility."""
    # Even gentler updates to avoid ratio explosions and KL pathologies
    base_cfg = {
        "learning_rate": 1e-6,
        "per_device_train_batch_size": 64,
        "num_mini_batches": 16,
        "num_ppo_epochs": 2,
        "gradient_accumulation_steps": 1,
        "cliprange": 0.2,
        "cliprange_value": 0.2,
        "vf_coef": 0.1,
        "seed": 42,
        "log_with": "wandb",
        "kl_coef": 0.001,
    }
    # Filter by constructor signature to avoid unexpected-arg errors
    allowed = set(inspect.signature(PPOConfig.__init__).parameters.keys())
    filtered = {k: v for k, v in base_cfg.items() if k in allowed}
    return PPOConfig(**filtered)


def load_model_and_tokenizer():
    """Load policy (with value head), reference model, and tokenizer."""
    #print("🤖 Loading Qwen2-0.5B-Instruct policy and reference models...")
    print("🤖 Loading LLaMA-3.2-3B-Instruct policy and reference models...")
    #print("🤖 Loading Qwen/Qwen2.5-3B-Instruct policy and reference models...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use bfloat16 for larger models to save memory while maintaining stability
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    #model_name = "Qwen/Qwen2-0.5B-Instruct"
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    #model_name = "Qwen/Qwen2.5-3B-Instruct"

    # Policy with value head
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else {"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        load_in_8bit=True,
    )

    # Reference model for KL control (with value head wrapper)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else {"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        load_in_8bit=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # decoder-only best practice for batching
        tokenizer.truncation_side = "left"  # ensure left truncation aligns with left padding

    # Ensure model configs have pad/eos ids set consistently
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    policy_model.config.eos_token_id = tokenizer.eos_token_id
    ref_model.config.eos_token_id = tokenizer.eos_token_id

    # Fix generation_config access for TRL compatibility
    from transformers import GenerationConfig
    if not hasattr(policy_model, 'generation_config'):
        policy_model.generation_config = GenerationConfig.from_model_config(policy_model.config)
    if not hasattr(ref_model, 'generation_config'):
        ref_model.generation_config = GenerationConfig.from_model_config(ref_model.config)

    print(f"✅ Models loaded on {device} | dtype={dtype}")
    return policy_model, ref_model, tokenizer


def setup_reward_model():
    """Setup reward model for training."""
    print("🎯 Setting up reward model...")
    
    reward_model = create_reward_model(top_k=100, reward_scale=1.0)
    
    # Quick smoke test
    test_query = "stem cell therapy pulmonary arterial hypertension"
    test_relevant_pmids = ["22776744", "25271670", "3493740"]
    _ = reward_model.compute_reward(test_query, test_relevant_pmids)
    print("✅ Reward model ready")
    return reward_model

def prepare_dataset_for_training(max_samples: int = 1000) -> Dataset:
    """Prepare dataset for training with shuffle and default 1000 samples."""
    print("📊 Loading dataset for training...")
    dataset = load_deepretrieval_dataset(
        data_path="data/train.jsonl", 
        max_samples=max_samples,
    )
    print(f"✅ Dataset loaded: {len(dataset)} examples")
    return dataset


def extract_boolean_query(output: str, prompt: str | None = None) -> str:
    """Extract and sanitize a single-line Boolean query from model output.
    Priority:
      0) If prompt is provided and present, drop everything up to and including the prompt
      1) Text inside <query>...</query>
      2) The LAST occurrence of "Boolean query:" in the text
      3) The first line after the last "Boolean query:" that contains AND/OR/NOT
      4) Fallback to the last line that contains AND/OR/NOT
    Then sanitize: strip SQL tails, normalize operators, collapse whitespace, cap length.
    """
    return output
    #if not output:
    #    return ""
    #text = output.strip()
#
    ## 0) Strip prompt prefix if available
    #if prompt:
    #    p = prompt.strip()
    #    if text.startswith(p):
    #        text = text[len(p):].lstrip()
    #    else:
    #        pos = text.find(p)
    #        if pos != -1:
    #            text = text[pos + len(p):].lstrip()
#
    ## Helper to decide if candidate is valid
    #def is_weak(s: str) -> bool:
    #    s_clean = s.strip().strip('()')
    #    if not s_clean:
    #        return True
    #    u = s_clean.upper()
    #    if u in {"AND", "OR", "NOT"}:
    #        return True
    #    return False
#
    ## 1) Tag-based extraction
    #candidate = None
    ##m = re.search(r"<query>(.*?)</query>", text, flags=re.IGNORECASE | re.DOTALL)
    ##tail_after_tag = None
    ##if m:
    ##    candidate = m.group(1).strip()
    ##    tail_after_tag = text[m.end():].strip()
    ##    if is_weak(candidate) and tail_after_tag:
    ##        # look in the tail for a stronger boolean line
    ##        for line in tail_after_tag.splitlines():
    ##            line = line.strip()
    ##            U = f" {line.upper()} "
    ##            if (" AND " in U) or (" OR " in U) or (" NOT " in U):
    ##                candidate = line
    ##                break
    ##if candidate is None:
    ##    # 2) Last "Boolean query:" occurrence
    ##    idx = text.lower().rfind("boolean query:")
    ##    if idx != -1:
    ##        after = text[idx + len("boolean query:") :].strip()
    ##        for line in after.splitlines():
    ##            line = line.strip()
    ##            if not line:
    ##                continue
    ##            candidate = line
    ##            break
    ### 3) If still none, take the last line with AND/OR/NOT
    ##if not candidate:
    ##    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    ##    for line in reversed(lines):
    ##        U = f" {line.upper()} "
    ##        if (" AND " in U) or (" OR " in U) or (" NOT " in U):
    ##            candidate = line
    ##            break
    ## 4) Final fallback: last non-empty line
    #if not candidate:
    #    candidate = text.splitlines()[0].strip()
    #    #lines_non_empty = [ln.strip() for ln in text.splitlines() if ln.strip()]
    #    #chosen = None
    #    #for ln in reversed(lines_non_empty):
    #    #    U = f" {ln.upper()} "
    #    #    if (" AND " in U) or (" OR " in U) or (" NOT " in U):
    #    #        chosen = ln
    #    #        break
    #    #if chosen is None:
    #    #    chosen = lines_non_empty[-1] if lines_non_empty else ""
    #    #candidate = chosen
#
    ## Remove trailing SQL or text after ';'
    #if ';' in candidate:
    #    candidate = candidate.split(';', 1)[0].strip()
    ## Remove obvious SQL keyword fragments
    #for token in ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "INSERT", "UPDATE", "DELETE"]:
    #    idx = candidate.upper().find(token)
    #    if idx != -1:
    #        candidate = candidate[:idx].strip()
    #        break
#
    ## Normalize boolean operators (case-insensitive)
    #def norm_ops(s: str) -> str:
    #    s = re.sub(r"\bAND\b", "AND", s, flags=re.IGNORECASE)
    #    s = re.sub(r"\bOR\b", "OR", s, flags=re.IGNORECASE)
    #    s = re.sub(r"\bNOT\b", "NOT", s, flags=re.IGNORECASE)
    #    return s
#
    #candidate = norm_ops(candidate)
#
    ## Collapse whitespace and quotes balance quick fix
    #candidate = " ".join(candidate.split())
    #if candidate.count('"') % 2 == 1:
    #    candidate = candidate.rstrip('"')
    #
    ## Cap to 40 tokens to keep queries concise
    #tokens = candidate.split()
    #if len(tokens) > 40:
    #    candidate = " ".join(tokens[:40])
#
    #return candidate


def tokenize_prompts(tokenizer, prompts: List[str], device: torch.device):
    return tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)


def run_full_training() -> bool:
    """Run PPO training with random mini-batches and real updates."""
    print("🚀 Phase 5: Full Training Pipeline")
    print("=" * 50)
    
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    #wandb.init(project="deepretrieval-ppo", name="phase5-full-training", config={
    #        "model": "Qwen/Qwen2-0.5B-Instruct",
    wandb.init(project="deepretrieval-ppo", name="llama3.2-3b-experiment", config={
            "model": "meta-llama/Llama-3.2-3B-Instruct",
    #wandb.init(project="deepretrieval-ppo", name="qwen2.5-3b-experiment", config={
    #        "model": "Qwen/Qwen2.5-3B-Instruct",
        "dataset_samples": 1000,
        "batch_size": 64,
        "updates": 600,
        "max_new_tokens": 32,
    })
    
    try:
        # Config and models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ppo_config = setup_ppo_config(device)
        policy_model, ref_model, tokenizer = load_model_and_tokenizer()
        dataset = prepare_dataset_for_training(max_samples=1000)
        
        # Tokenize dataset for TRL - add input_ids
        def tokenize_function(examples):
            # Tokenize the query text
            encoded = tokenizer(
                examples["query"],
                truncation=True,
                padding=False,  # Will be done by data collator
                max_length=512,
                return_tensors=None  # Return lists, not tensors
            )
            return encoded
        
        # Apply tokenization
        dataset = dataset.map(tokenize_function, batched=True)
        
        # Remove non-tensor columns that would confuse the data collator
        # Keep only the tokenizer outputs (input_ids, attention_mask, etc.)
        tokenizer_columns = ['input_ids', 'attention_mask']
        if hasattr(tokenizer, 'token_type_ids') and 'token_type_ids' in dataset.column_names:
            tokenizer_columns.append('token_type_ids')
        
        # Create a mapping from indices to original data for reward computation
        original_data = []
        for i, item in enumerate(dataset):
            original_data.append({
                "query": item["query"],
                "relevant_doc_ids": item["relevant_doc_ids"]
            })
        
        # Filter dataset to only include tokenizer columns
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in tokenizer_columns])
        
        reward_model = setup_reward_model()
        
        # Fix TRL compatibility issues with AutoModelForCausalLMWithValueHead
        if not hasattr(policy_model, 'is_gradient_checkpointing'):
            policy_model.is_gradient_checkpointing = True
        if not hasattr(ref_model, 'is_gradient_checkpointing'):
            ref_model.is_gradient_checkpointing = True

        # PPO trainer
        print("\n🔧 Initializing PPO trainer...")
        ppo_trainer = PPOTrainer(
            args=ppo_config,
            processing_class=tokenizer,
            reward_model=reward_model,
            train_dataset=dataset,
            model=policy_model,
            ref_model=ref_model,
            value_model=policy_model.pretrained_model,  # Use same model for both policy and value
        )
        print("✅ PPO trainer ready")
        
        # Training parameters 
        max_new_tokens = 32  
        temperature = 0.6
        top_p = 0.9

        print("\n🚀 Starting PPO training...")

        # Modern TRL Training Loop - Following Medium article pattern
        batch_count = 0
        for batch in tqdm(ppo_trainer.dataloader, desc="PPO Training"):
            # Get tokenized prompts (input_ids)
            query_tensors = batch["input_ids"]
            
            # Generate responses using PPO trainer's generate method
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            
            # Decode generated responses
            responses = [tokenizer.decode(output, skip_special_tokens=True) for output in response_tensors]
            
            # Extract boolean queries from responses for reward computation
            decoded_queries = [extract_boolean_query(resp) for resp in responses]
            
            # Get corresponding relevant docs for this batch
            batch_relevant = []
            for i in range(len(query_tensors)):
                data_idx = batch_count * len(query_tensors) + i
                if data_idx < len(original_data):
                    batch_relevant.append(original_data[data_idx]["relevant_doc_ids"])
                else:
                    batch_relevant.append([])  # Fallback
            
            # Compute rewards using the reward model
            rewards_list = reward_model.compute_rewards_batch(decoded_queries, batch_relevant)
            rewards = [torch.tensor(r, dtype=torch.float32) for r in rewards_list]
            
            # Perform PPO step - this is where the actual RL training happens
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            print(f"Batch {batch_count} - avg reward: {np.mean(rewards_list):.3f}")
            batch_count += 1
            
            # Optional: Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save final model and tokenizer
        #print("\n💾 Saving final model...")
        #save_dir = "models/phase5_final_recall"
        #os.makedirs(save_dir, exist_ok=True)
        #ppo_trainer.model.save_pretrained(save_dir)
        #tokenizer.save_pretrained(save_dir)
        #print(f"✅ Final model saved: {save_dir}")
        
        wandb.finish()
        print("\n🎉 Phase 5 completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Phase 5 failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            wandb.finish()
        except Exception:
            pass
        return False


def main():
    """Main function"""
    success = run_full_training()
    return success


if __name__ == "__main__":
    main() 