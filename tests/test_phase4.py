"""
Phase 4 Test: Dedicated Reward Model
Tests the new RetrievalRewardModel class with better separation of concerns
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.reward_model import RetrievalRewardModel, create_reward_model
from utils.data_utils import load_deepretrieval_dataset


def test_reward_model_creation():
    """Test 4.1: Create reward model"""
    print("🧪 Test 4.1: Creating reward model...")
    
    try:
        reward_model = create_reward_model(top_k=10, reward_scale=1.0)  # Changed from 5 to 10
        print("✅ Reward model created successfully")
        print(f"   Type: {type(reward_model)}")
        print(f"   Top-k: {reward_model.top_k}")
        print(f"   Reward scale: {reward_model.reward_scale}")
        return True
    except Exception as e:
        print(f"❌ Error creating reward model: {e}")
        return False


def test_single_reward_computation():
    """Test 4.2: Single query reward computation"""
    print("\n🧪 Test 4.2: Single query reward computation...")
    
    try:
        reward_model = create_reward_model()
        
        # Test query and ground truth
        test_query = "stem cell therapy pulmonary arterial hypertension"
        test_relevant_pmids = ["22776744", "25271670", "3493740"]
        
        reward = reward_model.compute_reward(test_query, test_relevant_pmids)
        
        print("✅ Single reward computation successful")
        print(f"   Query: '{test_query[:50]}...'")
        print(f"   Relevant PMIDs: {test_relevant_pmids[:3]}...")
        print(f"   Reward: {reward:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Error in single reward computation: {e}")
        return False


def test_batch_reward_computation():
    """Test 4.3: Batch reward computation"""
    print("\n🧪 Test 4.3: Batch reward computation...")
    
    try:
        reward_model = create_reward_model()
        
        # Test batch
        queries = [
            "stem cell therapy pulmonary arterial hypertension",
            "insulin resistance diabetes treatment",
            "statin preventive care"
        ]
        relevant_pmids_list = [
            ["22776744", "25271670", "3493740"],
            ["8532025", "10790348", "17504794"],
            ["10637197", "14967718", "17599437"]
        ]
        
        rewards = reward_model.forward(queries, relevant_pmids_list)
        
        print("✅ Batch reward computation successful")
        print(f"   Batch size: {len(queries)}")
        print(f"   Rewards shape: {rewards.shape}")
        print(f"   Rewards: {rewards.tolist()}")
        
        return True
    except Exception as e:
        print(f"❌ Error in batch reward computation: {e}")
        return False


def test_reward_statistics():
    """Test 4.4: Reward statistics tracking"""
    print("\n🧪 Test 4.4: Reward statistics tracking...")
    
    try:
        reward_model = create_reward_model()
        
        # Compute some rewards
        test_queries = [
            "stem cell therapy",
            "diabetes treatment",
            "preventive care"
        ]
        test_relevant_pmids_list = [
            ["22776744", "25271670"],
            ["8532025", "10790348"],
            ["10637197", "14967718"]
        ]
        
        for query, relevant_pmids in zip(test_queries, test_relevant_pmids_list):
            reward_model.compute_reward(query, relevant_pmids)
        
        # Get statistics
        stats = reward_model.get_stats()
        
        print("✅ Reward statistics tracking successful")
        print(f"   Total queries: {stats['total_queries']}")
        print(f"   Average reward: {stats['avg_reward']:.3f}")
        print(f"   History length: {len(stats['reward_history'])}")
        
        return True
    except Exception as e:
        print(f"❌ Error in reward statistics: {e}")
        return False


def test_reward_model_with_dataset():
    """Test 4.5: Reward model with actual dataset"""
    print("\n🧪 Test 4.5: Reward model with dataset...")
    
    try:
        # Load small dataset
        dataset = load_deepretrieval_dataset(data_path="data/train.jsonl", max_samples=3)
        reward_model = create_reward_model()
        
        print(f"   Dataset size: {len(dataset)}")
        
        # Test with dataset samples
        for i, item in enumerate(dataset):
            query = item["query"]
            relevant_pmids = item["relevant_doc_ids"]
            
            # Generate a simple test query (in real training, this would come from LLM)
            test_query = f"test query {i}: {query[:50]}"
            
            reward = reward_model.compute_reward(test_query, relevant_pmids)
            print(f"   Sample {i+1}: reward = {reward:.3f}")
        
        print("✅ Reward model works with dataset")
        return True
    except Exception as e:
        print(f"❌ Error with dataset: {e}")
        return False


def test_reward_model_configuration():
    """Test 4.6: Reward model configuration options"""
    print("\n🧪 Test 4.6: Reward model configuration...")
    
    try:
        # Test different configurations
        configs = [
            {"top_k": 3, "reward_scale": 0.5},
            {"top_k": 10, "reward_scale": 2.0},
            {"top_k": 10, "reward_scale": 1.0}  # Changed from 5 to 10
        ]
        
        for i, config in enumerate(configs):
            reward_model = create_reward_model(**config)
            print(f"   Config {i+1}: top_k={reward_model.top_k}, scale={reward_model.reward_scale}")
        
        print("✅ Reward model configuration successful")
        return True
    except Exception as e:
        print(f"❌ Error in configuration: {e}")
        return False


def main():
    """Run all Phase 4 tests"""
    print("🧪 Phase 4: Dedicated Reward Model")
    print("=" * 50)
    
    tests = [
        test_reward_model_creation,
        test_single_reward_computation,
        test_batch_reward_computation,
        test_reward_statistics,
        test_reward_model_with_dataset,
        test_reward_model_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🎉 Phase 4 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ Phase 4 completed successfully!")
        print("   Ready for Phase 5: Full training pipeline")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    main() 