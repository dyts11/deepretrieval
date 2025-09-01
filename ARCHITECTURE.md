# RL-Based Query Augmentation Architecture

## Overview

This document describes the architecture for a reinforcement learning-based query augmentation system that uses TRL's PPOTrainer to improve PubMed search query generation. The system takes PICO (Patients, Intervention, Comparison, Outcome) information as input and generates optimized PubMed search queries through RL training with real retrieval feedback.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PubMed        │    │  DialoGPT-small │    │   PubMed API    │
│   Dataset       │    │   (Policy)      │    │                 │
│                 │    │                 │    │                 │
│ PICO → Query    │───▶│ Generate        │───▶│ Search &        │
│ PMIDs (GT)      │    │ Response        │    │ Recall@5        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  PPO Update     │◀───│  Reward Signal  │
                       │                 │    │                 │
                       │ • Policy Loss   │    │ • 0.0 - 1.0     │
                       │ • Value Loss    │    │ • Based on      │
                       │ • KL Penalty    │    │   retrieval     │
                       └─────────────────┘    └─────────────────┘
```

## PPOTrainer Core Components

### 1. Policy Model (Actor)
**Purpose**: Generates responses from input queries\
**Implementation**: DialoGPT-small model\
**Integration Point**: Passed as `model` parameter to PPOTrainer
```python
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
```

**What PPOTrainer Does**:
- Manages model training state
- Handles gradient computation
- Applies PPO policy updates
- Manages model checkpointing

**What You Provide**:
- Base model selection
- Model configuration (device, dtype, etc.)

### 2. Value Model (Critic)
**Purpose**: Estimates value of states/actions for advantage computation\
**Implementation**: Same as policy model (DialoGPT-small)\
**Integration Point**: Passed as `value_model` parameter

**What PPOTrainer Does**:
- Computes value estimates
- Calculates advantages using GAE
- Updates value function
- Manages value loss computation

**What You Provide**:
- Value model instance (can be same as policy model)

### 3. Reference Model
**Purpose**: Provides baseline for KL divergence penalty\
**Implementation**: None (using same model as reference)\
**Integration Point**: Passed as `ref_model` parameter

**What PPOTrainer Does**:
- Computes KL divergence between current and reference policy
- Applies KL penalty to prevent large policy changes
- Manages reference model updates

**What You Provide**:
- Reference model (can be None to use same model)

### 4. Reward Model
**Purpose**: Computes reward signal for generated responses\
**Implementation**: PubMed API integration with recall@5 (reward function)\
**Integration Point**: Custom reward function integrated into training loop

**What PPOTrainer Does**:
- Calls reward function during training
- Manages reward computation timing
- Handles reward normalization

**What You Provide**:
```python
def compute_reward(generated_query, relevant_pmids):
    retrieved_pmids = pubmed_api.search(generated_query)
    recall = compute_recall(retrieved_pmids, relevant_pmids)
    return recall
```

### 5. Data Processing
**Purpose**: Handles dataset loading and preprocessing\
**Implementation**: DeepRetrieval dataset with PICO formatting\
**Integration Point**: Dataset passed as `train_dataset` parameter

**What PPOTrainer Does**:
- Manages DataLoader creation
- Handles batching and shuffling
- Manages dataset iteration
- Handles data collation

**What You Provide**:
```python
dataset = [
    {
        "query": "Generate PubMed query for PICO...",
        "relevant_doc_ids": ["12345", "67890"],
        "id": "sample_id"
    }
]
```

### 6. PPO Configuration
**Purpose**: Controls PPO algorithm hyperparameters\
**Implementation**: Custom PPOConfig\
**Integration Point**: Passed as `args` parameter

**What PPOTrainer Does**:
- Applies PPO hyperparameters
- Manages training loop configuration
- Handles learning rate scheduling
- Controls training duration

**What You Provide**:
```python
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=4,
    num_ppo_epochs=4,
    cliprange=0.2,
    # ... other parameters
)
```

## Integration Points

### 1. Reward Function Integration
**Location**: Training loop during reward computation
**Your Responsibility**:
- Implement PubMed API search
- Compute recall@5 metric
- Return reward value (0.0 to 1.0)

**PPOTrainer Responsibility**:
- Call reward function at appropriate times
- Handle reward computation errors
- Manage reward normalization

### 2. Dataset Format Integration
**Location**: Data loading and preprocessing
**Your Responsibility**:
- Format DeepRetrieval data to TRL format
- Ensure "query" column exists
- Include relevant PMIDs for reward computation

**PPOTrainer Responsibility**:
- Load and iterate through dataset
- Handle batching and shuffling
- Manage data collation

### 3. Model Integration
**Location**: Model initialization and training
**Your Responsibility**:
- Select appropriate base model
- Configure model parameters
- Handle model device placement

**PPOTrainer Responsibility**:
- Manage model training state
- Handle gradient computation
- Apply PPO updates
- Manage model checkpointing

## Implementation Details

### Data Flow
1. **Input**: PICO information from DeepRetrieval dataset
2. **Formatting**: Convert to query format for LLM
3. **Generation**: Policy model generates PubMed search query
4. **Evaluation**: Query used to search PubMed API
5. **Reward**: Recall@5 computed against ground truth PMIDs
6. **Update**: PPO algorithm updates model parameters

### Reward Computation
```python
def compute_reward(generated_query, relevant_pmids):
    # Search PubMed with generated query
    retrieved_pmids = pubmed_api.search_with_keywords(generated_query, topk=5)
    
    # Compute recall@5
    relevant_set = set(relevant_pmids)
    retrieved_set = set(retrieved_pmids)
    recall = len(retrieved_set & relevant_set) / len(relevant_set)
    
    return recall  # 0.0 to 1.0
```

### Training Configuration
```python
ppo_trainer = PPOTrainer(
    args=ppo_config,
    processing_class=tokenizer,
    model=policy_model,
    ref_model=None,
    reward_model=dummy_model,  # Not used, custom reward function
    train_dataset=dataset,
    value_model=value_model,
    data_collator=None
)
```

## What You Need to Implement

### 1. Custom Reward Function
- PubMed API integration
- Recall@5 computation
- Error handling for API failures

### 2. Data Preprocessing
- DeepRetrieval dataset loading
- PICO to query formatting
- Dataset validation

### 3. Model Configuration
- Base model selection (DialoGPT-small)
- Device placement (CPU/GPU)
- Model parameter configuration

### 4. Training Setup
- PPO hyperparameter tuning
- Training duration configuration
- Logging and monitoring setup

## What PPOTrainer Handles

### 1. PPO Algorithm Implementation
- Policy gradient computation
- Value function updates
- Advantage estimation (GAE)
- PPO clipping
- KL divergence penalties

### 2. Training Infrastructure
- Optimizer management
- Learning rate scheduling
- Gradient accumulation
- Model checkpointing
- Training state management

### 3. Data Management
- DataLoader creation
- Batch processing
- Dataset iteration
- Data collation

### 4. Logging and Monitoring
- Training metrics
- Loss computation
- Progress tracking
- Model evaluation

## Key Advantages of This Architecture

1. **Real-world Feedback**: PubMed API provides authentic evaluation
2. **End-to-end Learning**: Model learns to generate better queries
3. **Scalable**: Can work with any retrieval system
4. **Interpretable**: Reward directly measures retrieval performance
5. **Leverages TRL**: Uses battle-tested PPO implementation

## Implementation Phases

### Phase 1: Data Loading & Preprocessing ✅
- Load DeepRetrieval dataset
- Convert PICO format to query format
- Create TRL-compatible dataset

### Phase 2: PubMed API Integration ✅
- Implement reward computation
- Test API integration
- Validate reward function

### Phase 3: TRL PPO Setup ✅
- Configure PPOTrainer
- Test model generation
- Validate end-to-end pipeline

### Phase 4: Full Training Pipeline 🔄
- Optimize training performance
- Implement proper training loop
- Add logging and monitoring

### Phase 5: Evaluation & Optimization 📋
- Evaluate model performance
- Optimize hyperparameters
- Improve generation quality

This architecture allows the model to learn to generate better PubMed search queries through reinforcement learning with real retrieval feedback, while leveraging TRL's robust PPO implementation for stable training. 