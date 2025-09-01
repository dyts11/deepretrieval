#!/usr/bin/env python3
"""
PPO Training Script for Query Augmentation
Phase 3: TRL PPO Setup with Qwen2-0.5B-Instruct

This script implements the core PPO training loop for query augmentation
using TRL (Transformer Reinforcement Learning) framework.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
import inspect

from utils.data_utils import load_deepretrieval_dataset
from models.reward_model import create_reward_model


def setup_ppo_config():
    """Setup PPO configuration for testing with TRL-version-safe kwargs."""
    base_cfg = {
        "learning_rate": 1e-5,
        "batch_size": 1,
        "mini_batch_size": 1,
        "ppo_epochs": 1,
        "cliprange": 0.2,
        "cliprange_value": 0.2,
        "vf_coef": 0.1,
        "gamma": 1.0,
        "lam": 0.95,
        "kl_coef": 0.05,
        "bf16": False,
        "fp16": False,
        "seed": 42,
        "log_with": None,
    }
    allowed = set(inspect.signature(PPOConfig.__init__).parameters.keys())
    filtered = {k: v for k, v in base_cfg.items() if k in allowed}
    return PPOConfig(**filtered)


def setup_training_args():
    """Setup training arguments"""
    return TrainingArguments(
        output_dir="models/phase3_test",
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        save_strategy="epoch",
        logging_steps=5,
        report_to=None,  # No wandb for testing
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )


def load_model_and_tokenizer():
    """Load Qwen2-0.5B-Instruct value-head policy model and tokenizer"""
    print("🤖 Loading Qwen2-0.5B-Instruct model...")
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"🔄 Loading model on {device} with dtype {dtype}...")
    
    # Value-head wrapper required by PPOTrainer
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else "cpu",
        trust_remote_code=True
    )
    
    print(f"✅ Model loaded: {model_name}")
    try:
        print(f"   Parameters: {model.pretrained_model.num_parameters():,}")
    except Exception:
        pass
    print(f"   Device: {next(model.parameters()).device}")
    
    return model, tokenizer


def setup_reward_model():
    """Setup reward model using PubMed API"""
    print("🎯 Setting up reward model...")
    
    reward_model = create_reward_model(
        top_k=10, 
        reward_scale=1.0
    )
    
    print("🧪 Testing reward model...")
    test_query = "stem cell therapy pulmonary arterial hypertension"
    test_relevant_pmids = ["22776744", "25271670", "3493740"]
    _ = reward_model.compute_reward(test_query, test_relevant_pmids)
    print(f"✅ Reward model test successful")
    print("✅ Reward model ready")
    return reward_model


def get_model_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device('cpu')


def test_model_generation(model, tokenizer, dataset):
    print("\n🧪 Testing model generation...")
    sample = dataset[0]
    input_text = sample['query']
    print(f"Input: {input_text}")
    print(f"Input end")
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    backend = getattr(model, "pretrained_model", model)
    with torch.no_grad():
        outputs = backend.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    print(f"Generated End")
    return True


def setup_ppo_trainer(model, tokenizer, dataset, ppo_config, training_args):
    """Setup PPO trainer"""
    print("\n🔧 Setting up PPO trainer...")
    
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=None
    )
    print("✅ PPO trainer ready")
    return ppo_trainer


def run_training_loop(ppo_trainer, reward_model, dataset, max_episodes=1):
    """Run a single non-updating PPO pass: print raw outputs and rewards."""
    print(f"\n🚀 Starting evaluation loop (max {max_episodes} episode)...")
    try:
        queries = [item["query"] for item in dataset]
        device = next(ppo_trainer.model.parameters()).device
        print(f"Using device: {device}")
        print(f"\n📊 Processing {len(queries)} samples (no updates)...")
        print("=" * 80)
        query_tensors = [ppo_trainer.tokenizer(q, return_tensors="pt").input_ids.squeeze(0).to(device) for q in queries]
        response_tensors = []
        raw_outputs, extracted_outputs = [], []
        for i, query_tensor in enumerate(query_tensors):
            print(f"\n🔍 Sample {i+1}/{len(queries)}:")
            print(f"📝 INPUT (PICO prompt):")
            print(f"   {queries[i]}")
            with torch.no_grad():
                output_ids = ppo_trainer.model.generate(
                    input_ids=query_tensor.unsqueeze(0),
                    max_new_tokens=128,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7
                )[0]
            continuation_ids = output_ids[query_tensor.shape[-1]:]
            response_tensors.append(continuation_ids)
            raw_text = ppo_trainer.tokenizer.decode(continuation_ids, skip_special_tokens=True)
            raw_outputs.append(raw_text)
            extracted = extract_boolean_query(raw_text)
            extracted_outputs.append(extracted)
            print(f"📄 RAW OUTPUT:")
            print(f"   {raw_text}")
            print(f"🤖 EXTRACTED (first-line/boolean):")
            print(f"   {extracted}")
            print("-" * 80)
        print(f"\n🎯 Computing rewards for each sample:")
        print("=" * 80)
        rewards = []
        for i, extracted in enumerate(extracted_outputs):
            relevant_pmids = dataset[i]["relevant_doc_ids"]
            reward = reward_model.compute_reward(extracted, relevant_pmids)
            rewards.append(reward)
            print(f"Sample {i+1}: reward = {reward:.4f}")
            print(f"   Extracted query: {extracted}")
            print(f"   Relevant PMIDs: {relevant_pmids}")
            print("-" * 80)
        print(f"\n📊 Summary:")
        print(f"   Total samples processed: {len(rewards)}")
        if len(rewards) > 0:
            print(f"   Average reward: {sum(rewards)/len(rewards):.4f}")
            print(f"   Max reward: {max(rewards):.4f}")
            print(f"   Min reward: {min(rewards):.4f}")
        print(f"\n💡 Note: This is validation only - no model training occurred")
        return len(rewards)
    except Exception as e:
        print(f"❌ Error in evaluation loop: {e}")
        import traceback
        traceback.print_exc()
        return 0


def extract_boolean_query(output: str) -> str:
    return output.strip().splitlines()[0] if output else ""


def main():
    print("🧪 Phase 3: TRL PPO Setup with DialoGPT-small")
    print("=" * 60)
    try:
        model, tokenizer = load_model_and_tokenizer()
        print("\n📊 Loading dataset...")
        dataset = load_deepretrieval_dataset("data/train.jsonl", max_samples=5)
        print(f"✅ Dataset loaded: {len(dataset)} examples")
        reward_model = setup_reward_model()
        test_model_generation(model, tokenizer, dataset)
        ppo_config = setup_ppo_config()
        training_args = setup_training_args()
        ppo_trainer = setup_ppo_trainer(model, tokenizer, dataset, ppo_config, training_args)
        
        # Step 7: Run single evaluation loop (no updates)
        samples_processed = run_training_loop(ppo_trainer, reward_model, dataset, max_episodes=1)
        print(f"\n🎉 Phase 3 evaluation completed!")
        print(f"   Samples processed: {samples_processed}")
        print(f"   Ready for further testing or full training")
    except Exception as e:
        print(f"\n❌ Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True


if __name__ == "__main__":
    main() 