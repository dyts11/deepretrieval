#!/usr/bin/env python3
"""
Simple script to extract first 100 samples from DeepRetrieval dataset.
"""

import json
import os


def extract_first_100_samples(input_file: str = "data/train.jsonl", 
                            output_file: str = "data/sample_1000.jsonl") -> None:
    """
    Extract first 100 samples from the JSONL file.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to save the output JSONL file
    """
    print(f"📊 Reading from {input_file}...")
    
    samples = []
    count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if count >= 1000:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
                count += 1
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping invalid JSON line: {e}")
                continue
    
    print(f"✅ Read {len(samples)} samples")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Saved {len(samples)} samples to {output_file}")
    
    # Print sample information
    if samples:
        print(f"\n📋 Sample data preview:")
        print(f"   First sample ID: {samples[0]['id']}")
        print(f"   PICO keys: {list(samples[0]['pico'].keys())}")
        print(f"   Relevant docs count: {len(samples[0]['publication_pmids'])}")
        print(f"   Publication date: {samples[0]['pub_date']}")


def main():
    """Main function"""
    print("🔍 Extracting first 100 samples from DeepRetrieval dataset...")
    
    extract_first_100_samples()
    
    print(f"\n🎉 Extraction completed!")
    print(f"   Output file: data/sample_100.jsonl")
    print(f"   You can now use this file for faster testing")


if __name__ == "__main__":
    main() 