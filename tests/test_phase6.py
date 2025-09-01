"""
Phase 6 Test: Evaluation & Deployment
Tests the evaluation and deployment modules for Phase 6
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import Dict, List, Any

from evaluation.evaluator import ModelEvaluator, create_evaluator
from deployment.inference import QueryGenerator, QueryAPI, create_query_generator, create_query_api


def test_phase6_evaluator_creation():
    """Test 6.1: Model evaluator creation"""
    print("🧪 Test 6.1: Model evaluator creation...")
    
    try:
        # Use existing model for testing
        model_path = "models/phase3_test"
        evaluator = create_evaluator(model_path, max_samples=10)
        
        print("✅ Model evaluator creation successful")
        print(f"   Model path: {evaluator.model_path}")
        print(f"   Max samples: {evaluator.max_samples}")
        print(f"   Reward model type: {type(evaluator.reward_model)}")
        
        return True
    except Exception as e:
        print(f"❌ Error in evaluator creation: {e}")
        return False


def test_phase6_model_loading():
    """Test 6.2: Model loading in evaluator"""
    print("\n🧪 Test 6.2: Model loading in evaluator...")
    
    try:
        model_path = "models/phase3_test"
        evaluator = create_evaluator(model_path, max_samples=5)
        
        success = evaluator.load_model()
        
        if success:
            print("✅ Model loading successful")
            print(f"   Model type: {type(evaluator.model)}")
            print(f"   Tokenizer type: {type(evaluator.tokenizer)}")
            print(f"   Model parameters: {sum(p.numel() for p in evaluator.model.parameters()):,}")
        else:
            print("❌ Model loading failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error in model loading: {e}")
        return False


def test_phase6_dataset_preparation():
    """Test 6.3: Dataset preparation in evaluator"""
    print("\n🧪 Test 6.3: Dataset preparation...")
    
    try:
        model_path = "models/phase3_test"
        evaluator = create_evaluator(model_path, max_samples=5)
        
        dataset = evaluator.prepare_test_dataset()
        
        print("✅ Dataset preparation successful")
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Sample keys: {list(dataset[0].keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Error in dataset preparation: {e}")
        return False


def test_phase6_query_generation():
    """Test 6.4: Query generation in evaluator"""
    print("\n🧪 Test 6.4: Query generation...")
    
    try:
        model_path = "models/phase3_test"
        evaluator = create_evaluator(model_path, max_samples=3)
        evaluator.load_model()
        
        # Test prompts
        prompts = [
            "Generate a PubMed search query for this PICO information:\nP: Patients with PAH\nI: Stem cell therapy\nC: Standard treatment\nO: Efficacy",
            "Generate a PubMed search query for this PICO information:\nP: Diabetes patients\nI: Insulin therapy\nC: Placebo\nO: Blood glucose control"
        ]
        
        queries = evaluator.generate_queries(prompts)
        
        print("✅ Query generation successful")
        print(f"   Generated {len(queries)} queries")
        print(f"   Sample query: {queries[0][:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error in query generation: {e}")
        return False


def test_phase6_retrieval_evaluation():
    """Test 6.5: Retrieval performance evaluation"""
    print("\n🧪 Test 6.5: Retrieval performance evaluation...")
    
    try:
        model_path = "models/phase3_test"
        evaluator = create_evaluator(model_path, max_samples=5)
        evaluator.load_model()
        
        results = evaluator.evaluate_retrieval_performance()
        
        print("✅ Retrieval evaluation successful")
        print(f"   Average reward: {results['avg_reward']:.3f}")
        print(f"   Total queries: {results['total_queries']}")
        print(f"   Reward distribution: {results['reward_distribution']}")
        
        return True
    except Exception as e:
        print(f"❌ Error in retrieval evaluation: {e}")
        return False


def test_phase6_quality_evaluation():
    """Test 6.6: Query quality evaluation"""
    print("\n🧪 Test 6.6: Query quality evaluation...")
    
    try:
        model_path = "models/phase3_test"
        evaluator = create_evaluator(model_path, max_samples=5)
        evaluator.load_model()
        
        results = evaluator.evaluate_query_quality()
        
        print("✅ Quality evaluation successful")
        print(f"   Average query length: {results['avg_query_length']:.1f}")
        print(f"   Repetitive queries: {results['repetitive_queries']}")
        print(f"   Empty queries: {results['empty_queries']}")
        
        return True
    except Exception as e:
        print(f"❌ Error in quality evaluation: {e}")
        return False


def test_phase6_report_generation():
    """Test 6.7: Evaluation report generation"""
    print("\n🧪 Test 6.7: Evaluation report generation...")
    
    try:
        model_path = "models/phase3_test"
        evaluator = create_evaluator(model_path, max_samples=5)
        evaluator.load_model()
        
        report = evaluator.generate_evaluation_report()
        
        print("✅ Report generation successful")
        print(f"   Report length: {len(report)} characters")
        print(f"   Contains retrieval metrics: {'Average Reward' in report}")
        print(f"   Contains quality metrics: {'Average Length' in report}")
        
        return True
    except Exception as e:
        print(f"❌ Error in report generation: {e}")
        return False


def test_phase6_deployment_generator():
    """Test 6.8: Deployment query generator"""
    print("\n🧪 Test 6.8: Deployment query generator...")
    
    try:
        model_path = "models/phase3_test"
        generator = create_query_generator(model_path)
        
        # Test PICO input
        pico_info = {
            'P': 'Patients with advanced pulmonary arterial hypertension (PAH)',
            'I': 'Stem cell therapy',
            'C': 'Standard drug treatment or placebo',
            'O': 'Efficacy of stem cell therapy for PAH'
        }
        
        result = generator.generate_query(pico_info)
        
        print("✅ Deployment generator successful")
        print(f"   Generated query: {result['generated_query'][:50]}...")
        print(f"   Has timestamp: {'timestamp' in result}")
        print(f"   Has generation params: {'generation_params' in result}")
        
        return True
    except Exception as e:
        print(f"❌ Error in deployment generator: {e}")
        return False


def test_phase6_deployment_api():
    """Test 6.9: Deployment API wrapper"""
    print("\n🧪 Test 6.9: Deployment API wrapper...")
    
    try:
        model_path = "models/phase3_test"
        api = create_query_api(model_path)
        
        # Test single generation
        pico_info = {
            'P': 'Diabetes patients',
            'I': 'Insulin therapy',
            'C': 'Placebo',
            'O': 'Blood glucose control'
        }
        
        result = api.generate_single(pico_info)
        
        print("✅ Deployment API successful")
        print(f"   Single generation: {result['generated_query'][:50]}...")
        
        # Test batch generation
        pico_list = [pico_info, pico_info]
        batch_results = api.generate_batch(pico_list)
        
        print(f"   Batch generation: {len(batch_results)} results")
        
        return True
    except Exception as e:
        print(f"❌ Error in deployment API: {e}")
        return False


def test_phase6_visualization():
    """Test 6.10: Visualization capabilities"""
    print("\n🧪 Test 6.10: Visualization capabilities...")
    
    try:
        model_path = "models/phase3_test"
        evaluator = create_evaluator(model_path, max_samples=5)
        evaluator.load_model()
        
        # Test plotting (without showing)
        evaluator.plot_results()
        
        print("✅ Visualization successful")
        print("   Plots generated successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error in visualization: {e}")
        return False


def main():
    """Run all Phase 6 tests"""
    print("🧪 Phase 6: Evaluation & Deployment")
    print("=" * 50)
    
    tests = [
        test_phase6_evaluator_creation,
        test_phase6_model_loading,
        test_phase6_dataset_preparation,
        test_phase6_query_generation,
        test_phase6_retrieval_evaluation,
        test_phase6_quality_evaluation,
        test_phase6_report_generation,
        test_phase6_deployment_generator,
        test_phase6_deployment_api,
        test_phase6_visualization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🎉 Phase 6 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ Phase 6 completed successfully!")
        print("   Evaluation and deployment modules ready")
        print("   Ready for production deployment")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    main() 