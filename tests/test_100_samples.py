#!/usr/bin/env python3
"""
Quick test to verify all 100 samples work without API errors
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_deepretrieval_dataset
from models.reward_model import create_reward_model
import time

def test_100_samples():
    """Test all 100 samples to ensure they work without API errors"""
    print("🧪 Testing all 100 samples for API compatibility...")
    print("=" * 60)
    
    try:
        # Load all 100 samples
        print("📊 Loading 100 samples...")
        dataset = load_deepretrieval_dataset("data/train.jsonl", max_samples=100)
        print(f"✅ Loaded {len(dataset)} samples")
        
        # Create reward model
        print("🎯 Setting up reward model...")
        reward_model = create_reward_model()
        print("✅ Reward model ready")
        
        # Test all samples
        print(f"\n🔍 Testing {len(dataset)} samples...")
        
        start_time = time.time()
        
        # Process in batches of 20 (same as full training)
        batch_size = 20
        all_rewards = []
        
        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_samples = dataset.select(range(i, batch_end))
            
            print(f"  Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size} ({len(batch_samples)} samples)...")
            
            # Simulate model-generated responses (like full training does)
            # For testing, we'll use simple PICO-based queries instead of the full prompt
            batch_responses = []
            batch_relevant_pmids = batch_samples["relevant_doc_ids"]
            
            for sample in batch_samples:
                # Extract PICO information from the prompt
                query_text = sample["query"]
                
                # Extract PICO components (simplified for testing)
                if "P: " in query_text and "I: " in query_text:
                    # Extract PICO parts
                    p_start = query_text.find("P: ") + 3
                    p_end = query_text.find("I: ")
                    p_part = query_text[p_start:p_end].strip()
                    
                    i_start = query_text.find("I: ") + 3
                    i_end = query_text.find("C: ")
                    i_part = query_text[i_start:i_end].strip()
                    
                    # Create a simple search query from PICO
                    response = f"{p_part} AND {i_part}"
                else:
                    # Fallback: use a simple medical query
                    response = "medical research"
                
                batch_responses.append(response)
            
            # Compute rewards in batch using the simulated responses
            batch_rewards = reward_model.compute_rewards_batch(batch_responses, batch_relevant_pmids)
            all_rewards.extend(batch_rewards)
            
            print(f"    ✅ Batch completed: {len(batch_rewards)} rewards computed")
        
        total_time = time.time() - start_time
        
        # Results
        print(f"\n📊 Test Results:")
        print(f"  ✅ All {len(dataset)} samples processed successfully")
        print(f"  ⏱️  Total time: {total_time:.2f}s")
        print(f"  🚀 Rate: {len(dataset)/total_time:.1f} samples/second")
        print(f"  📈 Average reward: {sum(all_rewards)/len(all_rewards):.3f}")
        
        # Performance prediction for full training
        time_per_episode = total_time
        total_episodes = 100
        total_training_time = time_per_episode * total_episodes / 3600  # hours
        
        print(f"\n🎯 Full Training Prediction:")
        print(f"  Time per episode: {time_per_episode/60:.1f} minutes")
        print(f"  Total training time: {total_training_time:.1f} hours")
        print(f"  Total API calls: {len(dataset) * total_episodes:,}")
        
        print(f"\n✅ All samples tested successfully!")
        print(f"   Ready for full training with {total_episodes} episodes")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_100_samples()
    if success:
        print("\n🎉 Test passed! Full training should work without API errors.")
    else:
        print("\n❌ Test failed. Please check the issues before running full training.") 