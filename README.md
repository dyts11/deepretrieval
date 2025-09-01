# My DeepRetrieval (TRL-based RL Query Augmentation Framework)

## Overview
A reinforcement learning-based query augmentation framework using **TRL (Transformer Reinforcement Learning)** from Hugging Face. It uses PPO (Proximal Policy Optimization) with GAE (Generalized Advantage Estimation) to train a language model to rewrite queries for improved PubMed retrieval performance.

**Key Features:**
- Uses TRL's battle-tested PPO implementation
- BM25-based retrieval reward model
- Simple and modular design
- Easy to extend to other datasets and tasks

---

## Architecture

### Core Components

1. **PPO Trainer** (`trl.PPOTrainer`): Handles the RL training loop with proper GAE, advantage normalization, and policy clipping
2. **Reward Model**: BM25-based retrieval performance evaluator
3. **Language Model**: Generates rewritten queries (e.g., DialoGPT-small)
4. **BM25 Retriever**: Simple but effective retrieval for reward computation

### Workflow

```
Original Query → Language Model → Rewritten Query → BM25 Retrieval → Reward → PPO Update
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
python train_ppo.py
```

This will:
- Create sample data automatically
- Initialize a small DialoGPT model
- Train with PPO using retrieval-based rewards
- Log metrics to Weights & Biases
- Save the trained model

---

## Configuration

### Model Settings
- **Base Model**: `microsoft/DialoGPT-small` (small for demo)
- **Generation**: Max length 32, temperature 0.7, top-p 0.9
- **Training**: Learning rate 1e-5, batch size 4

### PPO Parameters
- **Clip Range**: 0.2 (standard PPO clipping)
- **GAE Lambda**: 0.95 (advantage estimation)
- **Value Coefficient**: 0.1 (value function loss weight)
- **KL Coefficient**: 0.05 (KL divergence penalty)

### Reward Function
- **Retrieval**: BM25 with top-5 documents
- **Metric**: Recall@5 (fraction of relevant docs in top-5)
- **Range**: 0.0 to 1.0

---

## Data Format

### Training Data (`data/train.jsonl`)
```json
{"query": "diabetes treatment", "relevant_doc_ids": ["doc1", "doc2", "doc3"]}
{"query": "cancer prevention", "relevant_doc_ids": ["doc4", "doc5"]}
```

### Corpus Data (`data/pubmed_corpus.json`)
```json
[
  {"doc_id": "doc1", "text": "Diabetes treatment involves managing blood sugar..."},
  {"doc_id": "doc2", "text": "Insulin therapy is a common treatment..."}
]
```

---

## Advantages of TRL Approach

### vs. Custom Implementation
1. **Battle-tested**: Used in production RLHF systems
2. **Optimized**: Proper GAE, advantage normalization, memory management
3. **Rich ecosystem**: Integrates with Hugging Face tools
4. **Active maintenance**: Regular updates and bug fixes
5. **Built-in features**: Logging, checkpointing, distributed training

### Key Benefits
- **GAE**: Proper advantage estimation with λ=0.95
- **Clipping**: Prevents large policy updates (cliprange=0.2)
- **Value Function**: Separate critic for better training stability
- **KL Penalty**: Prevents policy from diverging too much
- **Memory Optimization**: Efficient handling of large models

---

## Monitoring Training

### Weights & Biases Metrics
- `objective/rlhf_reward`: Main objective (should increase)
- `val/ratio`: Policy change ratio (should be ~1.0)
- `policy/entropy`: Policy randomness
- `loss/policy_avg`: Policy loss
- `loss/value_avg`: Value function loss

### Debugging Tips
- **Reward not increasing**: Check reward function, data quality
- **Ratio too high/low**: Adjust learning rate or cliprange
- **Memory issues**: Reduce batch size or use gradient accumulation

---

## Extending the Framework

### Custom Reward Models
```python
class CustomRewardModel:
    def compute_reward(self, query: str, relevant_doc_ids: List[str]) -> float:
        # Your custom reward logic
        return reward
```

### Different Base Models
```python
model_name = "gpt2"  # or any other causal LM
```

### Alternative Retrieval Methods
- Dense retrieval (sentence-transformers)
- Hybrid retrieval (BM25 + dense)
- Neural retrieval models

---

## References

- [TRL Documentation](https://huggingface.co/docs/trl/ppo_trainer)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [GAE Paper](https://arxiv.org/abs/1506.02438)
- [DeepRetrieval Paper](https://arxiv.org/abs/2204.12741)

---

## Example Output

```
Episode 1: Reward = 0.333
Episode 2: Reward = 0.500
Episode 3: Reward = 0.667
...
Episode 10: Reward = 0.800
Model saved to models/query_augmentation_ppo
```

The model learns to rewrite queries to improve retrieval performance, as measured by recall@5 on the relevant documents. 