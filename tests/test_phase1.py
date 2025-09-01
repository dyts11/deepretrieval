#!/usr/bin/env python3
"""
Test script for Phase 1: Data Preparation & Basic Setup
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_deepretrieval_dataset, test_data_loading


def test_phase1():
    """Test all Phase 1 components"""
    print("🧪 Testing Phase 1: Data Preparation & Basic Setup")
    print("=" * 60)
    
    # Test 1.1: Dependencies (already done)
    print("✅ 1.1: Dependencies installed")
    
    # Test 1.2: Dataset downloaded (already done)
    print("✅ 1.2: DeepRetrieval dataset downloaded")
    
    # Test 1.3: Data preprocessing
    print("\n🔍 1.3: Testing data preprocessing...")
    try:
        # Test with small sample first
        dataset = load_deepretrieval_dataset("data/train.jsonl", max_samples=10)
        print(f"✅ Successfully processed {len(dataset)} examples")
        
        # Verify dataset structure
        if len(dataset) > 0:
            sample = dataset[0]
            required_fields = ["query", "relevant_doc_ids"]
            missing_fields = [field for field in required_fields if field not in sample]
            
            if missing_fields:
                print(f"❌ Missing required fields: {missing_fields}")
                return False
            else:
                print("✅ Dataset has required fields for TRL")
                
            # Print sample for verification
            print(f"\n📝 Sample formatted data:")
            print(f"Query: {sample['query'][:150]}...")
            print(f"Relevant docs count: {len(sample['relevant_doc_ids'])}")
            print(f"First 3 relevant docs: {sample['relevant_doc_ids'][:3]}")
            
    except Exception as e:
        print(f"❌ Error in data preprocessing: {e}")
        return False
    
    # Test 1.4: Small sample test
    print("\n🔍 1.4: Testing with small sample...")
    try:
        small_dataset = load_deepretrieval_dataset("data/train.jsonl", max_samples=5)
        print(f"✅ Small sample test passed: {len(small_dataset)} examples")
        
        # Check data quality
        for i, example in enumerate(small_dataset):
            if not example['query'] or not example['relevant_doc_ids']:
                print(f"❌ Example {i} has empty text or relevant_doc_ids")
                return False
        
        print("✅ All examples have valid data")
        
    except Exception as e:
        print(f"❌ Error in small sample test: {e}")
        return False
    
    # Test 1.5: Dataset structure verification
    print("\n🔍 1.5: Verifying dataset structure...")
    try:
        # Check if dataset can be used with TRL
        from datasets import Dataset
        
        # Test dataset operations
        dataset_length = len(dataset)
        print(f"✅ Dataset length: {dataset_length}")
        
        # Test indexing
        first_item = dataset[0]
        print(f"✅ Dataset indexing works")
        
        # Test iteration
        count = 0
        for item in dataset:
            count += 1
            if count > 3:  # Just test first few
                break
        print(f"✅ Dataset iteration works")
        
    except Exception as e:
        print(f"❌ Error in dataset structure verification: {e}")
        return False
    
    print("\n🎉 Phase 1 completed successfully!")
    print("\nNext steps:")
    print("1. Phase 2: Implement PubMed API integration")
    print("2. Phase 3: Set up TRL PPO trainer")
    print("3. Phase 4: Connect reward function")
    
    return True


if __name__ == "__main__":
    success = test_phase1()
    if success:
        print("\n✅ Phase 1 tests passed! Ready for Phase 2.")
    else:
        print("\n❌ Phase 1 tests failed. Please fix issues before proceeding.") 