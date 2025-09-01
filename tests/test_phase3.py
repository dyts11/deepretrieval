#!/usr/bin/env python3
"""
Test script for Phase 3: TRL PPO Setup with DialoGPT-small
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from datasets import Dataset

from utils.data_utils import load_deepretrieval_dataset
from utils.pubmed_api import create_pubmed_retriever

import traceback    

def test_phase3():
    """Test all Phase 3 components"""
    print("🧪 Testing Phase 3: TRL PPO Setup with DialoGPT-small")
    print("=" * 60)
    
    # Test 3.1: Model loading
    print("\n🔍 3.1: Testing model loading...")
    try:
        model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Force CPU usage to avoid MPS memory issues
        print("🔄 Loading model on CPU to avoid memory issues...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        print(f"✅ Model loaded on CPU")
        
        print(f"   Model: {model_name}")
        print(f"   Parameters: {model.num_parameters():,}")
        print(f"   Device: {next(model.parameters()).device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Test 3.2: Dataset compatibility
    print("\n🔍 3.2: Testing dataset compatibility...")
    try:
        dataset = load_deepretrieval_dataset("data/train.jsonl", max_samples=5)
        print(f"✅ Dataset loaded: {len(dataset)} examples")
        
        # Test tokenization
        sample = dataset[0]
        inputs = tokenizer(sample['query'], return_tensors="pt", truncation=True, max_length=128)
        print(f"✅ Tokenization works: {inputs['input_ids'].shape}")
        
    except Exception as e:
        print(f"❌ Error with dataset: {e}")
        return False
    
    # Test 3.3: Model generation
    print("\n🔍 3.3: Testing model generation...")
    try:
        # Create a more explicit prompt to encourage generation
        base_text = dataset[0]['query']
        sample_text = base_text + "\n\nQuery: "
        
        inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=128)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # Generate more new tokens
                temperature=1.2,     # Higher temperature for more creativity
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Slight repetition penalty
                no_repeat_ngram_size=3   # Prevent repeating 3-grams
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if generation actually happened
        if len(generated_text) > len(sample_text):
            print(f"✅ Generation successful")
            print(f"   Input: {sample_text[:100]}...")
            print(f"   Output: {generated_text}")
            print(f"   Generated {len(generated_text) - len(sample_text)} additional characters")
        else:
            print(f"⚠️  Generation may not have worked properly")
            print(f"   Input: {sample_text[:100]}...")
            print(f"   Output: {generated_text}")
            print(f"   No additional text generated")
        
    except Exception as e:
        print(f"❌ Error in generation: {e}")
        return False
    
    # Test 3.4: PPO configuration
    print("\n🔍 3.4: Testing PPO configuration...")
    try:
        ppo_config = PPOConfig(
            learning_rate=1e-5,
            batch_size=1,  # Set batch size to 1 for testing
            mini_batch_size=1,
            num_ppo_epochs=2,
            total_episodes=10,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            gamma=1.0,
            lam=0.95,
            kl_coef=0.05,
            bf16=False,  # Disable bf16 for CPU
            fp16=False,   # Disable fp16 for CPU
        )
        
        print(f"✅ PPO configuration created")
        print(f"   Learning rate: {ppo_config.learning_rate}")
        print(f"   Batch size: {ppo_config.batch_size}")
        
    except Exception as e:
        print(f"❌ Error in PPO configuration: {e}")
        return False
    
    # Test 3.5: PPO trainer setup
    print("\n🔍 3.5: Testing PPO trainer setup...")
    try:
        ppo_trainer = PPOTrainer(
            args=ppo_config,
            processing_class=tokenizer,
            model=model,
            ref_model=None,
            reward_model=model,  # Dummy
            train_dataset=dataset,
            value_model=model,   # Dummy
            data_collator=None
        )
        print(f"✅ PPO trainer created successfully")
        
    except Exception as e:
        print(f"❌ Error creating PPO trainer: {e}")
        traceback.print_exc()
        return False
    
    # Test 3.6: Basic training step
    print("\n🔍 3.6: Testing basic training step...")
    print(f"Dataset: {dataset}")
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset length: {len(dataset)}") 
    print(f"Dataset schema: {dataset.features}")
    print(f"Dataset column names: {dataset.column_names}")
    tokens = tokenizer(dataset[0]["query"], return_tensors="pt")
    print("Sample tokenized query:", tokens)

    try:
        # Get queries as strings
        queries = [item["query"] for item in dataset]

        # Tokenize queries
        query_tensors = [tokenizer(q, return_tensors="pt").input_ids.squeeze(0).to(ppo_trainer.model.policy.device) for q in queries]

        # Generate responses from the model
        response_tensors = []
        for query_tensor in query_tensors:
            output = ppo_trainer.model.policy.generate(
                input_ids=query_tensor.unsqueeze(0),
                max_new_tokens=32,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
            # Remove the prompt part to get only the new tokens
            response = output[0][query_tensor.shape[-1]:]
            response_tensors.append(response)

        # Decode to string for reward computation
        responses = [tokenizer.decode(resp, skip_special_tokens=True) for resp in response_tensors]

        print(type(ppo_trainer.model))
        # Compute rewards using your function
        pubmed_api = create_pubmed_retriever()
        rewards = []
        for i, response in enumerate(responses):
            relevant_pmids = dataset[i]["relevant_doc_ids"]
            reward = pubmed_api.compute_reward(response, relevant_pmids)
            rewards.append(reward)

        rewards = [torch.tensor(reward, dtype=torch.float32) for reward in rewards]
        
        # # Test the actual PPO training using train() method
        # print("Testing PPOTrainer.train()...")
        # 
        # # Configure for a short training run
        # ppo_trainer.args.num_train_epochs = 1
        # ppo_trainer.args.per_device_train_batch_size = 1
        # ppo_trainer.args.max_steps = 2  # Very short for testing
        # 
        # # Run training
        # train_result = ppo_trainer.train()
        # 
        # print(f"✅ PPO training completed successfully")
        # print(f"   Training loss: {train_result.training_loss:.4f}")
        # print(f"   Global step: {train_result.global_step}")

    except Exception as e:
        print(f"❌ Error in training step: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3.7: Reward model integration
    print("\n🔍 3.7: Testing reward model integration...")
    try:
        pubmed_api = create_pubmed_retriever()
        
        # Test with a simple query
        test_query = "diabetes treatment"
        test_pmids = ["12345678", "87654321"]  # Dummy PMIDs
        
        reward = pubmed_api.compute_reward(test_query, test_pmids)
        
        print(f"✅ Reward computation works: {reward:.3f}")
        
    except Exception as e:
        print(f"❌ Error in reward integration: {e}")
        return False
    
    print("\n🎉 Phase 3 completed successfully!")
    print("\nNext steps:")
    print("1. Phase 4: Connect real reward function to training loop")
    print("2. Phase 5: Implement full training pipeline")
    print("3. Phase 6: Evaluation and optimization")
    
    return True


def test_model_quality():
    """Test the quality of DialoGPT-small for medical queries"""
    print("\n🔍 Testing DialoGPT-small quality for medical queries...")
    
    try:
        # Load model
        model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Force CPU usage to avoid MPS memory issues
        print("🔄 Loading model on CPU for quality test...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",
            low_cpu_mem_usage=True
        )

        # # Check if GPU is available
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # print(f"Using device: {device}, dtype: {dtype}")
        # 
        # model = AutoModelForCausalLM.from_pretrained(
        #     "microsoft/DialoGPT-small",
        #     device_map="auto" if torch.cuda.is_available() else "cpu",  # ← CHANGED
        #     torch_dtype=dtype,  # ← CHANGED
        #     low_cpu_mem_usage=True
        # )
        # tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        
        # Test medical queries
        test_queries = [
            "Generate a PubMed search query for this PICO information:\nP: Patients with diabetes\nI: Insulin therapy\nC: Standard treatment\nO: Efficacy",
            "Generate a PubMed search query for this PICO information:\nP: Cancer patients\nI: Chemotherapy\nC: Radiation therapy\nO: Survival rates"
        ]
        
        for i, query in enumerate(test_queries):
            print(f"\nTest query {i+1}:")
            print(f"Input: {query[:100]}...")
            
            inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,  # Generate new tokens
                    temperature=0.8,     # Higher temperature
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2  # Prevent repetition
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated}")
            
            # Check if generation actually happened
            if len(generated) > len(query):
                print("✅ New text generated")
            else:
                print("⚠️  No new text generated")
            
            # Simple quality check
            if "diabetes" in query.lower() and "diabetes" in generated.lower():
                print("✅ Relevant medical terms found")
            elif "cancer" in query.lower() and ("cancer" in generated.lower() or "chemotherapy" in generated.lower()):
                print("✅ Relevant medical terms found")
            else:
                print("⚠️  Medical terms not clearly present")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model quality: {e}")
        return False


if __name__ == "__main__":
    # Test basic functionality
    success = test_phase3()
    
    if success:
        # Test model quality
        test_model_quality()
        
        print("\n✅ Phase 3 tests passed! Ready for Phase 4.")
    else:
        print("\n❌ Phase 3 tests failed. Please fix issues before proceeding.") 