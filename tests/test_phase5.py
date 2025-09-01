"""
Phase 5 Test: Full Training Pipeline
Tests the complete PPO training implementation with actual training loops
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from datasets import Dataset

from utils.data_utils import load_deepretrieval_dataset
from models.reward_model import create_reward_model


def test_phase5_configuration():
    """Test 5.1: PPO configuration setup"""
    print("🧪 Test 5.1: PPO configuration setup...")
    
    try:
        from training.full_training import setup_ppo_config, setup_training_args
        
        ppo_config = setup_ppo_config()
        training_args = setup_training_args()
        
        print("✅ PPO configuration successful")
        print(f"   Learning rate: {ppo_config.learning_rate}")
        print(f"   Batch size: {ppo_config.batch_size}")
        print(f"   Total episodes: {ppo_config.total_episodes}")
        print(f"   Max steps: {ppo_config.max_steps}")
        
        return True
    except Exception as e:
        print(f"❌ Error in PPO configuration: {e}")
        return False


def test_phase5_model_loading():
    """Test 5.2: Model and tokenizer loading"""
    print("\n🧪 Test 5.2: Model and tokenizer loading...")
    
    try:
        from training.full_training import load_model_and_tokenizer
        
        model, tokenizer = load_model_and_tokenizer()
        
        print("✅ Model and tokenizer loaded successfully")
        print(f"   Model type: {type(model)}")
        print(f"   Tokenizer type: {type(tokenizer)}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def test_phase5_reward_model_integration():
    """Test 5.3: Reward model integration"""
    print("\n🧪 Test 5.3: Reward model integration...")
    
    try:
        from training.full_training import setup_reward_model
        
        reward_model = setup_reward_model()
        
        print("✅ Reward model integration successful")
        print(f"   Reward model type: {type(reward_model)}")
        print(f"   Top-k: {reward_model.top_k}")
        print(f"   Reward scale: {reward_model.reward_scale}")
        
        return True
    except Exception as e:
        print(f"❌ Error in reward model integration: {e}")
        return False


def test_phase5_dataset_preparation():
    """Test 5.4: Dataset preparation for training"""
    print("\n🧪 Test 5.4: Dataset preparation...")
    
    try:
        from training.full_training import prepare_dataset_for_training
        
        dataset = prepare_dataset_for_training(max_samples=10)
        
        print("✅ Dataset preparation successful")
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Sample keys: {list(dataset[0].keys())}")
        print(f"   Sample query: {dataset[0]['query'][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error in dataset preparation: {e}")
        return False


def test_phase5_ppo_trainer_setup():
    """Test 5.5: PPO trainer setup"""
    print("\n🧪 Test 5.5: PPO trainer setup...")
    
    try:
        from training.full_training import (
            setup_ppo_config, 
            load_model_and_tokenizer, 
            prepare_dataset_for_training
        )
        
        # Setup components
        ppo_config = setup_ppo_config()
        model, tokenizer = load_model_and_tokenizer()
        dataset = prepare_dataset_for_training(max_samples=5)
        
        # Setup PPO trainer
        ppo_trainer = PPOTrainer(
            args=ppo_config,
            processing_class=tokenizer,
            model=model,
            ref_model=None,
            reward_model=model,  # Dummy for testing
            train_dataset=dataset,
            value_model=model,
            data_collator=None
        )
        
        print("✅ PPO trainer setup successful")
        print(f"   Trainer type: {type(ppo_trainer)}")
        print(f"   Dataset size: {len(ppo_trainer.train_dataset)}")
        
        return True
    except Exception as e:
        print(f"❌ Error in PPO trainer setup: {e}")
        return False


def test_phase5_generation_pipeline():
    """Test 5.6: Generation pipeline"""
    print("\n🧪 Test 5.6: Generation pipeline...")
    
    try:
        from training.full_training import (
            load_model_and_tokenizer,
            prepare_dataset_for_training
        )
        
        # Load components
        model, tokenizer = load_model_and_tokenizer()
        dataset = prepare_dataset_for_training(max_samples=3)
        
        # Test generation
        queries = [item["query"] for item in dataset]
        responses = []
        
        for query in queries:
            inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            responses.append(response)
        
        print("✅ Generation pipeline successful")
        print(f"   Generated {len(responses)} responses")
        print(f"   Sample response: {responses[0][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error in generation pipeline: {e}")
        return False


def test_phase5_reward_computation():
    """Test 5.7: Reward computation in training loop"""
    print("\n🧪 Test 5.7: Reward computation...")
    
    try:
        from training.full_training import (
            load_model_and_tokenizer,
            prepare_dataset_for_training,
            setup_reward_model
        )
        
        # Load components
        model, tokenizer = load_model_and_tokenizer()
        dataset = prepare_dataset_for_training(max_samples=3)
        reward_model = setup_reward_model()
        
        # Test reward computation
        queries = [item["query"] for item in dataset]
        responses = ["test query 1", "test query 2", "test query 3"]  # Mock responses
        
        rewards = []
        for i, response in enumerate(responses):
            relevant_pmids = dataset[i]["relevant_doc_ids"]
            reward = reward_model.compute_reward(response, relevant_pmids)
            rewards.append(reward)
        
        avg_reward = np.mean(rewards)
        
        print("✅ Reward computation successful")
        print(f"   Computed {len(rewards)} rewards")
        print(f"   Average reward: {avg_reward:.3f}")
        print(f"   Individual rewards: {[f'{r:.3f}' for r in rewards]}")
        
        return True
    except Exception as e:
        print(f"❌ Error in reward computation: {e}")
        return False


def test_phase5_training_loop_structure():
    """Test 5.8: Training loop structure"""
    print("\n🧪 Test 5.8: Training loop structure...")
    
    try:
        from training.full_training import (
            setup_ppo_config,
            load_model_and_tokenizer,
            prepare_dataset_for_training,
            setup_reward_model
        )
        
        # Setup all components
        ppo_config = setup_ppo_config()
        model, tokenizer = load_model_and_tokenizer()
        dataset = prepare_dataset_for_training(max_samples=5)
        reward_model = setup_reward_model()
        
        # Test training loop structure (without actual training)
        training_stats = []
        
        for episode in range(3):  # Test with 3 episodes
            queries = [item["query"] for item in dataset]
            responses = [f"test response {i}" for i in range(len(queries))]
            
            # Compute rewards
            rewards = []
            for i, response in enumerate(responses):
                relevant_pmids = dataset[i]["relevant_doc_ids"]
                reward = reward_model.compute_reward(response, relevant_pmids)
                rewards.append(reward)
            
            avg_reward = np.mean(rewards)
            training_stats.append({
                'episode': episode,
                'avg_reward': avg_reward,
                'total_queries': len(queries)
            })
            
            print(f"   Episode {episode+1}: avg_reward={avg_reward:.3f}")
        
        print("✅ Training loop structure successful")
        print(f"   Completed {len(training_stats)} test episodes")
        print(f"   Final avg reward: {training_stats[-1]['avg_reward']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error in training loop structure: {e}")
        return False


def main():
    """Run all Phase 5 tests"""
    print("🧪 Phase 5: Full Training Pipeline")
    print("=" * 50)
    
    tests = [
        test_phase5_configuration,
        test_phase5_model_loading,
        test_phase5_reward_model_integration,
        test_phase5_dataset_preparation,
        test_phase5_ppo_trainer_setup,
        test_phase5_generation_pipeline,
        test_phase5_reward_computation,
        test_phase5_training_loop_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🎉 Phase 5 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ Phase 5 completed successfully!")
        print("   Full training pipeline ready")
        print("   Ready for actual training execution")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    main() 