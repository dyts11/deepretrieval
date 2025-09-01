 ### Model.eval()
 The model.eval() method is used to set the model to evaluation mode. This is important because certain layers in a neural network, such as dropout and batch normalization, behave differently during training and evaluation.

### Why need to use eval mode for generation
- **Dropout:** During training, dropout randomly “drops out” (sets to zero) a fraction of the input units to prevent overfitting. In evaluation mode, all units are used, and dropout is effectively turned off. **That’s fine during training (adds regularization), but terrible for generation, text becomes random noise.**
- **Batch Normalization:** During training, batch normalization calculates statistics (mean and variance) based on the mini - batch. In evaluation mode, it uses the running statistics (accumulated during training) to normalize the input.
- **Generation cache disabled:** Hugging Face sets config.use_cache=False during training for efficiency (so it can do gradient checkpointing). Without the cache, autoregressive generation can behave differently or inefficiently. If you try to generate() with train() mode + use_cache=False, you’ll sometimes see garbled outputs.

### torch.no_grad()
torch.no_grad() impacts the autograd engine and deactivate it. It will **reduce memory usage** and **speed up computations** but you won’t be able to backprop (which you don’t want in an eval script).
The torch.no_grad() context manager is used to **disable gradient calculation**. Gradient calculation is a computationally expensive operation that is primarily used during the training phase to update the model’s parameters using backpropagation.

### PPO Trainer Workflow
```python
# 1. Load tokenizer and models
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. PPO configuration
ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    batch_size=4,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
)

# 3. Create PPOTrainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=None
)

# 4. Define reward function
def compute_reward(query, response):
    return ...

# 5. Training loop
for epoch in range(3):  # training epochs
    for query in queries:
        # Encode query
        query_tensors = tokenizer(query, return_tensors="pt", padding=True).input_ids
        
        # switch to evaluation mode
        ppo_trainer.model.eval() 
        # Generate response
        with torch.no_grad(): # don't need gradient for generating response
            response_tensors = policy_model.generate(
                query_tensors,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode response
        response_texts = tokenizer.batch_decode(response_tensors[:, query_tensors.shape[1]:], skip_special_tokens=True)

        # Compute rewards
        rewards = [compute_reward(query, response_texts[0])]

        # Run PPO step 
        # where the gradient are actually compute
        # switch to model.train()
        ppo_trainer.step([query_tensors[0]], [response_tensors[0]], rewards) 


        