import json
from typing import List, Dict, Any
from datasets import Dataset


def load_pubmed_data(data_path: str) -> List[Dict[str, Any]]:
    """Load PubMed dataset from JSONL file"""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_pico_for_llm(pico_dict: Dict[str, str]) -> str:
    """Convert PICO dict to LLM input format using few-shot Boolean query prompt"""
    user_prompt = f'''Create a PubMed Boolean search query:

EXAMPLE (format only - do not use this content):
Boolean query: ("diabetes" AND "insulin therapy" AND "blood glucose control")

Patient: {pico_dict.get('P', '')}
Intervention: {pico_dict.get('I', '')}
Comparison: {pico_dict.get('C', '')}
Outcome: {pico_dict.get('O', '')}

IMPORTANT: Output ONLY the Boolean query in the format shown above. Do NOT include any explanations, notes, or extra text. Your response must be a single line containing only the Boolean query.

Boolean query:'''
    return user_prompt


def prepare_deepretrieval_data(data: List[Dict[str, Any]]) -> Dataset:
    """Convert DeepRetrieval dataset format to TRL format"""
    formatted_data = []
    
    for item in data:
        # Extract PICO information
        if "pico" not in item:
            continue
            
        pico = item["pico"]
        
        # Create input text for LLM
        input_text = format_pico_for_llm(pico)
        
        # Extract relevant document IDs
        relevant_doc_ids = item.get("publication_pmids", [])
        
        # Create formatted item for TRL
        formatted_item = {
            "query": input_text,  # Input to LLM
            "relevant_doc_ids": relevant_doc_ids,  # Ground truth for reward
            "id": item.get("id", ""),  # Keep original ID for reference
        }
        
        formatted_data.append(formatted_item)
    
    return Dataset.from_list(formatted_data)


def load_deepretrieval_dataset(data_path: str, max_samples: int = 100) -> Dataset:
    """Load and format DeepRetrieval dataset for TRL training"""
    print(f"Loading DeepRetrieval dataset from {data_path}")
    
    # Load raw data
    data = load_pubmed_data(data_path)
    print(f"Loaded {len(data)} raw examples")
    
    # Limit samples for testing (default to 100)
    if max_samples:
        data = data[:max_samples]
        print(f"Limited to {len(data)} samples for testing")
    
    # Convert to TRL format
    dataset = prepare_deepretrieval_data(data)
    print(f"Converted to {len(dataset)} TRL-formatted examples")
    
    # Print sample for verification
    if len(dataset) > 0:
        print("\nSample formatted data:")
        sample = dataset[0]
        print(f"Query: {sample['query'][:200]}...")
        print(f"Relevant docs: {sample['relevant_doc_ids'][:3]}...")
        print(f"ID: {sample['id']}")
    
    return dataset


def test_data_loading():
    """Test the data loading and preprocessing"""
    print("Testing DeepRetrieval data loading...")
    
    try:
        # Try to load the actual dataset
        dataset = load_deepretrieval_dataset("data/train.jsonl", max_samples=5)
        print(f"✅ Successfully loaded {len(dataset)} examples")
        
        # Print first example details
        if len(dataset) > 0:
            first_example = dataset[0]
            print(f"\nFirst example:")
            print(f"Query length: {len(first_example['query'])}")
            print(f"Relevant docs count: {len(first_example['relevant_doc_ids'])}")
            print(f"Query preview: {first_example['query'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


if __name__ == "__main__":
    # Test data loading
    test_data_loading() 