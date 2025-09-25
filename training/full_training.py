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
    # Silence the repetitive gradient-checkpointing vs caching warnings from transformers
    hf_logging.set_verbosity_error()
    warnings.filterwarnings(
        "ignore",
        message="Caching is incompatible with gradient checkpointing",
        category=UserWarning,
        module="transformers",
    )
    
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
        reward_model = setup_reward_model()
        
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
        num_updates = 600
        batch_size = 64
        max_new_tokens = 32  
        temperature = 0.6
        top_p = 0.9

        indices = list(range(len(dataset)))
        device_obj = next(ppo_trainer.model.parameters()).device

        print("\n🚀 Starting PPO updates...")
        progress = tqdm(range(num_updates), desc="PPO Updates")

        for update_idx in progress:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Sample a random mini-batch
            batch_indices = random.sample(indices, batch_size)
            batch_prompts = [dataset[i]["query"] for i in batch_indices]
            batch_relevant = [dataset[i]["relevant_doc_ids"] for i in batch_indices]
                    
            # Build query tensors as list for TRL API
            query_tensors = [
                tokenizer(p, return_tensors="pt", truncation=True, max_length=512).input_ids.squeeze(0).to(device_obj)
                for p in batch_prompts
            ]
                    
            ppo_trainer.model.eval()
            # Use TRL's generate helper to ensure consistent masks/logprobs for KL
            with torch.no_grad():
                response_tensors = ppo_trainer.generate(
                    query_tensors,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                    
            # Slice continuations and decode relative to prompts
            cont_tensors = []
            decoded_responses = []
            for q_ids, r_ids, prompt in zip(query_tensors, response_tensors, batch_prompts):
                input_len = q_ids.shape[-1]
                cont = r_ids[input_len:]
                cont_tensors.append(cont)
                raw = tokenizer.decode(cont, skip_special_tokens=True)
                decoded_responses.append(raw)

            response_tensors = cont_tensors

            # Filter invalid/empty continuations and off-format outputs prior to PPO
            #def is_valid_text(s: str) -> bool:
            #    if not s:
            #        return False
            #    U = f" {s.upper()} "
            #    has_bool = (" AND " in U) or (" OR " in U) or (" NOT " in U)
            #    ascii_ratio = sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))
            #    return has_bool and (ascii_ratio >= 0.8)

            #mask = [ (c.numel() > 0) and is_valid_text(t) for c, t in zip(response_tensors, decoded_responses) ]
            #if not any(mask):
            #    # Skip this update if nothing valid
            #    continue
            ## Apply mask consistently
            #query_tensors = [q for q, m in zip(query_tensors, mask) if m]
            #response_tensors = [c for c, m in zip(response_tensors, mask) if m]
            #decoded_responses = [t for t, m in zip(decoded_responses, mask) if m]
            #batch_relevant = [r for r, m in zip(batch_relevant, mask) if m]

            # Compute rewards (batch)
            rewards_list = reward_model.compute_rewards_batch(decoded_responses, batch_relevant)
            avg_reward = float(np.mean(rewards_list)) if len(rewards_list) > 0 else 0.0
            # TRL expects scores as list of torch tensors
            scores = [torch.tensor(r, dtype=torch.float32, device=device_obj) for r in rewards_list]

            # PPO update step
            stats = ppo_trainer.step(query_tensors, response_tensors, scores)
            
            # Extract metrics (handle numpy arrays)
            def extract_metric(stats_dict, key, default=0.0):
                value = stats_dict.get(key, default)
                if hasattr(value, '__iter__') and not isinstance(value, str):
                    try:
                        return float(value[0]) if len(value) > 0 else default
                    except (IndexError, TypeError):
                        return default
                return float(value) if isinstance(value, (int, float)) else default
            
            # Core PPO metrics
            #kl_value = extract_metric(stats, "objective/kl")
            kl_value = float(stats.get("objective/kl", 0.0))
            policy_loss_value = extract_metric(stats, "ppo/loss/policy")
            value_loss_value = extract_metric(stats, "ppo/loss/value")
            total_loss_value = extract_metric(stats, "ppo/loss/total")

            # Policy evaluation metrics
            entropy_value = extract_metric(stats, "ppo/policy/entropy")
            approx_kl_value = extract_metric(stats, "ppo/policy/approxkl")
            clip_fraction = extract_metric(stats, "ppo/policy/clipfrac")
            advantages_mean = extract_metric(stats, "ppo/policy/advantages_mean")
            
            # Value function metrics
            value_error = extract_metric(stats, "ppo/val/error")
            value_clip_fraction = extract_metric(stats, "ppo/val/clipfrac")
            explained_variance = extract_metric(stats, "ppo/val/var_explained")
            
            # Reward and score metrics
            mean_scores = float(stats.get("ppo/mean_scores", 0.0))
            std_scores = float(stats.get("ppo/std_scores", 0.0))
            
            # Token length metrics
            query_len_mean = extract_metric(stats, "tokens/queries_len_mean")
            response_len_mean = extract_metric(stats, "tokens/responses_len_mean")
            
            # Log comprehensive metrics
            wandb.log({
                # Core training metrics
                "train/update": update_idx,
                "train/avg_reward": avg_reward,
                "train/kl": kl_value,
                "train/policy_loss": policy_loss_value,
                "train/value_loss": value_loss_value,
                "train/total_loss": total_loss_value,
                
                # Policy evaluation
                "policy/entropy": entropy_value,
                "policy/approx_kl": approx_kl_value,
                "policy/clip_fraction": clip_fraction,
                "policy/advantages_mean": advantages_mean,
                
                # Value function evaluation
                "value/prediction_error": value_error,
                "value/clip_fraction": value_clip_fraction,
                "value/explained_variance": explained_variance,
                
                # Reward analysis
                "ppo/mean_scores": mean_scores,
                "ppo/std_scores": std_scores,
                
                # Generation analysis
                "generation/query_length": query_len_mean,
                "generation/response_length": response_len_mean,
            })
            progress.set_postfix({
                "avg_reward": f"{avg_reward:.3f}", 
                "kl": f"{kl_value:.3f}", 
                "policy_loss": f"{policy_loss_value:.3f}", 
                "value_loss": f"{value_loss_value:.3f}"
            })

            # Optional: free GPU cache periodically
            if torch.cuda.is_available() and (update_idx + 1) % 10 == 0:
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