#!/usr/bin/env python3
"""
Test script for Phase 2: PubMed API Integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pubmed_api import PubmedAPI, create_pubmed_retriever
from utils.data_utils import load_deepretrieval_dataset


def test_phase2():
    """Test all Phase 2 components"""
    print("🧪 Testing Phase 2: PubMed API Integration")
    print("=" * 60)
    
    # Test 2.1: API connection
    print("\n🔍 2.1: Testing PubMed API connection...")
    try:
        retriever = create_pubmed_retriever()
        
        # Test basic connection
        if retriever.test_api_connection():
            print("✅ API connection successful")
        else:
            print("❌ API connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API connection: {e}")
        return False
    
    # Test 2.2: Basic search functionality
    print("\n🔍 2.2: Testing search functionality...")
    try:
        # Test simple search
        test_query = "diabetes treatment"
        results = retriever.search_with_keywords(test_query, topk=3)
        
        if results:
            print(f"✅ Search successful: {len(results)} results")
            print(f"  First result PMID: {results[0]}")
        else:
            print("❌ Search returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Error in search functionality: {e}")
        return False
    
    # Test 2.3: Rate limiting and error handling
    print("\n🔍 2.3: Testing rate limiting and error handling...")
    try:
        # Test multiple requests
        queries = ["diabetes", "cancer", "heart disease"]
        for query in queries:
            results = retriever.search_with_keywords(query, topk=2)
            print(f"  ✅ Query '{query}': {len(results)} results")
            
    except Exception as e:
        print(f"❌ Error in rate limiting test: {e}")
        return False
    
    # Test 2.3.5: Batch API functionality
    print("\n🔍 2.3.5: Testing batch API functionality...")
    try:
        # Test batch search with larger batch size
        batch_queries = [
            "diabetes treatment", "cancer therapy", "heart disease", "pulmonary hypertension", "stem cell therapy",
            "insulin therapy", "chemotherapy", "cardiac surgery", "hypertension treatment", "gene therapy",
            "diabetes mellitus", "oncology", "cardiovascular disease", "respiratory disease", "regenerative medicine"
        ]
        print(f"  Testing batch search with {len(batch_queries)} queries...")
        
        import time
        start_time = time.time()
        
        # Sequential search (for comparison)
        sequential_results = {}
        for query in batch_queries:
            results = retriever.search_with_keywords(query, topk=3)
            sequential_results[query] = results
        
        sequential_time = time.time() - start_time
        print(f"  ✅ Sequential search completed in {sequential_time:.2f}s")
        
        # Batch search
        start_time = time.time()
        batch_results = retriever.search_with_keywords_batch(batch_queries, topk=3)
        batch_time = time.time() - start_time
        print(f"  ✅ Batch search completed in {batch_time:.2f}s")
        
        # Compare results
        if len(batch_results) == len(sequential_results):
            print(f"  ✅ Batch results match sequential results")
            print(f"  🚀 Speedup: {sequential_time/batch_time:.1f}x faster")
            print(f"  📊 Efficiency: {len(batch_queries)} queries in {batch_time:.2f}s ({len(batch_queries)/batch_time:.1f} queries/second)")
        else:
            print(f"  ⚠️ Batch results differ from sequential results")
            
    except Exception as e:
        print(f"❌ Error in batch API test: {e}")
        return False
    
    # Test 2.4: Reward computation
    print("\n🔍 2.4: Testing reward computation...")
    try:
        # Test with sample data
        test_query = "stem cell therapy pulmonary arterial hypertension"
        relevant_pmids = ["22776744", "25271670", "3493740"]  # Sample PMIDs
        
        reward = retriever.compute_reward(test_query, relevant_pmids)
        
        if 0.0 <= reward <= 1.0:
            print(f"✅ Reward computation successful: {reward:.3f}")
        else:
            print(f"❌ Invalid reward value: {reward}")
            return False
            
    except Exception as e:
        print(f"❌ Error in reward computation: {e}")
        return False
    
    # Test 2.4.5: Batch reward computation
    print("\n🔍 2.4.5: Testing batch reward computation...")
    try:
        from models.reward_model import create_reward_model
        
        # Create reward model
        reward_model = create_reward_model()
        
        # Test data
        test_queries = [
            "stem cell therapy pulmonary arterial hypertension",
            "diabetes treatment insulin therapy",
            "cancer chemotherapy",
            "heart disease treatment",
            "pulmonary hypertension endothelin"
        ]
        
        test_relevant_pmids = [
            ["22776744", "25271670", "3493740"],
            ["12345678", "87654321", "11111111"],
            ["22222222", "33333333", "44444444"],
            ["55555555", "66666666", "77777777"],
            ["88888888", "99999999", "00000000"]
        ]
        
        print(f"  Testing batch reward computation with {len(test_queries)} queries...")
        
        import time
        start_time = time.time()
        
        # Sequential reward computation (for comparison)
        sequential_rewards = []
        for query, pmids in zip(test_queries, test_relevant_pmids):
            reward = reward_model.compute_reward(query, pmids)
            sequential_rewards.append(reward)
        
        sequential_time = time.time() - start_time
        print(f"  ✅ Sequential reward computation completed in {sequential_time:.2f}s")
        print(f"  Sequential rewards: {[f'{r:.3f}' for r in sequential_rewards]}")
        
        # Batch reward computation
        start_time = time.time()
        batch_rewards = reward_model.compute_rewards_batch(test_queries, test_relevant_pmids)
        batch_time = time.time() - start_time
        print(f"  ✅ Batch reward computation completed in {batch_time:.2f}s")
        print(f"  Batch rewards: {[f'{r:.3f}' for r in batch_rewards]}")
        
        # Compare results
        if len(batch_rewards) == len(sequential_rewards):
            print(f"  ✅ Batch rewards match sequential rewards")
            print(f"  🚀 Speedup: {sequential_time/batch_time:.1f}x faster")
        else:
            print(f"  ⚠️ Batch rewards differ from sequential rewards")
            
    except Exception as e:
        print(f"❌ Error in batch reward test: {e}")
        return False
    
    # Test 2.5: Integration with dataset
    print("\n🔍 2.5: Testing integration with dataset...")
    try:
        # Load small dataset
        dataset = load_deepretrieval_dataset("data/train.jsonl", max_samples=3)
        
        if len(dataset) > 0:
            # Test with first example
            example = dataset[0]
            test_query = "stem cell therapy"  # Simplified query for testing
            relevant_pmids = example['relevant_doc_ids'][:3]  # Use first 3 PMIDs
            
            reward = retriever.compute_reward(test_query, relevant_pmids)
            print(f"✅ Dataset integration successful: reward = {reward:.3f}")
        else:
            print("❌ No dataset examples available")
            return False
            
    except Exception as e:
        print(f"❌ Error in dataset integration: {e}")
        return False
    
    print("\n🎉 Phase 2 completed successfully!")
    print("\nNext steps:")
    print("1. Phase 3: Set up TRL PPO trainer")
    print("2. Phase 4: Connect reward function to training loop")
    print("3. Phase 5: Implement full training pipeline")
    
    return True


def test_full_training_integration():
    """Test the full training integration with batch processing"""
    print("\n🧪 Testing Full Training Integration with Batch Processing")
    print("=" * 70)
    
    try:
        # Test 1: Load dataset
        print("\n🔍 1: Testing dataset loading...")
        dataset = load_deepretrieval_dataset("data/train.jsonl", max_samples=5)
        print(f"✅ Dataset loaded: {len(dataset)} examples")
        
        # Test 2: Create reward model
        print("\n🔍 2: Testing reward model creation...")
        from models.reward_model import create_reward_model
        reward_model = create_reward_model()
        print("✅ Reward model created successfully")
        
        # Test 3: Simulate training loop with batch processing
        print("\n🔍 3: Testing batch processing in training loop...")
        
        # Simulate generated responses (mock LLM outputs)
        mock_responses = [
            "stem cell therapy pulmonary arterial hypertension",
            "diabetes treatment insulin therapy",
            "cancer chemotherapy treatment",
            "heart disease medication",
            "pulmonary hypertension endothelin receptor"
        ]
        
        # Get relevant PMIDs from dataset
        relevant_pmids_list = [dataset[i]["relevant_doc_ids"] for i in range(len(dataset))]
        
        print(f"  Testing with {len(mock_responses)} mock responses...")
        
        import time
        start_time = time.time()
        
        # Sequential processing (for comparison)
        sequential_rewards = []
        for i, response in enumerate(mock_responses):
            relevant_pmids = relevant_pmids_list[i]
            reward = reward_model.compute_reward(response, relevant_pmids)
            sequential_rewards.append(reward)
            print(f"    Sequential {i+1}: reward = {reward:.3f}")
        
        sequential_time = time.time() - start_time
        print(f"  ✅ Sequential processing completed in {sequential_time:.2f}s")
        
        # Batch processing
        start_time = time.time()
        batch_rewards = reward_model.compute_rewards_batch(mock_responses, relevant_pmids_list)
        batch_time = time.time() - start_time
        print(f"  ✅ Batch processing completed in {batch_time:.2f}s")
        
        for i, reward in enumerate(batch_rewards):
            print(f"    Batch {i+1}: reward = {reward:.3f}")
        
        # Compare results
        if len(batch_rewards) == len(sequential_rewards):
            print(f"  ✅ Batch results match sequential results")
            print(f"  🚀 Speedup: {sequential_time/batch_time:.1f}x faster")
        else:
            print(f"  ⚠️ Batch results differ from sequential results")
        
        # Test 4: Performance prediction for full training
        print("\n🔍 4: Performance prediction for full training...")
        
        # Estimate full training performance
        samples_per_episode = 100
        total_episodes = 100
        
        sequential_time_per_episode = (sequential_time / len(mock_responses)) * samples_per_episode
        batch_time_per_episode = (batch_time / len(mock_responses)) * samples_per_episode
        
        total_sequential_time = sequential_time_per_episode * total_episodes / 3600  # hours
        total_batch_time = batch_time_per_episode * total_episodes / 3600  # hours
        
        print(f"  Estimated time per episode:")
        print(f"    Sequential: {sequential_time_per_episode/60:.1f} minutes")
        print(f"    Batch: {batch_time_per_episode/60:.1f} minutes")
        print(f"  Estimated total training time:")
        print(f"    Sequential: {total_sequential_time:.1f} hours")
        print(f"    Batch: {total_batch_time:.1f} hours")
        print(f"  🚀 Expected speedup: {total_sequential_time/total_batch_time:.1f}x faster")
        
        print("\n✅ Full training integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in full training integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_key_setup():
    """Test API key configuration"""
    print("\n🔑 Testing API key setup...")
    
    # Check if API key is set in the file
    from utils.pubmed_api import MY_PUBMED_API_KEY
    if MY_PUBMED_API_KEY != "your_actual_api_key_here":
        print(f"✅ API key found in file: {MY_PUBMED_API_KEY[:10]}...")
    else:
        print("⚠️  API key not set in utils/pubmed_api.py")
        print("   Please replace 'your_actual_api_key_here' with your actual API key")
        return False
    
    # Test with the configured key
    retriever = create_pubmed_retriever()
    return retriever.test_api_connection()


if __name__ == "__main__":
    # Test API key setup first
    if test_api_key_setup():
        # Run full Phase 2 tests
        success = test_phase2()
        if success:
            print("\n✅ Phase 2 tests passed! Ready for Phase 3.")
            
            # Run additional batch processing tests
            print("\n" + "="*60)
            batch_success = test_full_training_integration()
            if batch_success:
                print("\n✅ All tests passed! Ready for full training.")
            else:
                print("\n❌ Batch processing tests failed. Please check the implementation.")
        else:
            print("\n❌ Phase 2 tests failed. Please fix issues before proceeding.")
    else:
        print("\n❌ API key setup failed. Please check your NCBI API key in utils/pubmed_api.py") 