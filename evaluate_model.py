#!/usr/bin/env python3
"""
Simple evaluation script for the trained DeepRetrieval model
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluation.evaluator import create_evaluator

def main():
    """Main evaluation function"""
    
    # Path to your trained model
    model_path = "models/phase3_test"
    
    print("🔍 Starting model evaluation...")
    print(f"📁 Model path: {model_path}")
    
    # Create evaluator
    evaluator = create_evaluator(
        model_path=model_path,
        max_samples=50  # Evaluate on 50 samples for quick testing
    )
    
    # Load model
    if not evaluator.load_model():
        print("❌ Failed to load model. Exiting.")
        return
    
    # Prepare test dataset
    evaluator.prepare_test_dataset()
    
    print("\n📊 Running evaluations...")
    
    # Run retrieval performance evaluation
    print("1️⃣ Evaluating retrieval performance...")
    retrieval_results = evaluator.evaluate_retrieval_performance()
    
    print(f"   Average Reward: {retrieval_results['avg_reward']:.3f}")
    print(f"   Reward Std: {retrieval_results['std_reward']:.3f}")
    print(f"   Max Reward: {retrieval_results['max_reward']:.3f}")
    print(f"   Min Reward: {retrieval_results['min_reward']:.3f}")
    
    # Run query quality evaluation
    print("\n2️⃣ Evaluating query quality...")
    quality_results = evaluator.evaluate_query_quality()
    
    print(f"   Average Query Length: {quality_results['avg_query_length']:.1f} words")
    print(f"   Repetitive Queries: {quality_results['repetitive_queries']}")
    print(f"   Empty Queries: {quality_results['empty_queries']}")
    
    # Generate comprehensive report
    print("\n3️⃣ Generating evaluation report...")
    report = evaluator.generate_evaluation_report(save_path="evaluation_results.md")
    
    print("\n📈 Creating visualization plots...")
    evaluator.plot_results(save_path="evaluation_plots.png")
    
    print("\n✅ Evaluation complete!")
    print("📄 Report saved to: evaluation_results.md")
    print("📊 Plots saved to: evaluation_plots.png")
    
    # Print summary
    print("\n" + "="*50)
    print("📋 EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {model_path}")
    print(f"Average Reward: {retrieval_results['avg_reward']:.3f}")
    print(f"Query Quality: {quality_results['avg_query_length']:.1f} avg words")
    print("="*50)

if __name__ == "__main__":
    main() 