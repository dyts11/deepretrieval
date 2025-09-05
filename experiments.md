### Experiment setup
#### ppo config:
learning_rate: 1e-5              How fast the model learns
batch_size: 16,                  Number of samples per update
mini_batch_size: 4               Subset size for gradient updates
ppo_epochs: 1                    How many times to reuse each batch
gradient_accumulation_steps: 1   Steps before applying gradients
cliprange: 0.2                   Policy update clipping threshold
cliprange_value: 0.2             Value function update clipping
vf_coef: 0.1                     Weight of value loss in total loss
target_kl: 0.1                   KL divergence stopping criterion
seed: 42                         Reproducibility
log_with: "wandb"                Logging backend
adap_kl_ctrl: True               Adaptive KL penalty
init_kl_coef: 1                  Initial KL penalty coefficient

#### generation parameters
num_updates = 600
batch_size = 16
max_new_tokens = 32  
temperature = 0.7
top_p = 0.9

#### reward
length of generated query, the shorter the respond, the higher the reward
```
q = query.split()
length = len(q)
reward = 2 - length/16
```

### experiment 1
#### training description
Set low target kl (0.1), and very strick kl penalty (1).
Use output length as reward.

#### training results
![](graph/train_1_train.png
)
![](graph/train_1_policy.png)

#### Detailed Analysis

**Reward Optimization Success:**
- Clear upward trend from ~0.5 to ~1.8
- Shows the model is successfully learning to generate shorter responses

**KL Divergence Control - CRITICAL FAILURE:**
- Negative KL warning for 122/600 updates: UserWarning: KL divergence is starting to become negative: -30.55 - this might be a precursor for failed training. sometimes this happens because the generation kwargs are not correctly set. Please make sure that the generation kwargs are set correctly, or review your training hyperparameters.
- This is highly unusual behaviour - mathematically impossible, indicates numerical instability
- Extreme high KL values (20-300) far exceeding target_kl (0.1)
- Policy changing drastically from reference model
- Adaptive KL control mechanism completely failing

**Value Function Instability - MAJOR ISSUE:**
- Massive spikes reaching 3000+ around steps 300-400
- This is extremely unstable and indicates the value function is completely failing to predict returns
- Value function cannot adapt to rapidly changing reward distribution
- Causes unreliable advantage estimates, leading to poor policy updates

**Policy Loss Volatility:**
- Shows significant spikes (up to 1.2) around the same period
- More volatile than ideal, but not as severe as value loss
- Indicates noisy policy updates

**Entropy Collapse - EXPLORATION LOSS:**
- Rapid drop from ~11 to near 0 and stays there
- Model becomes too deterministic too quickly
- Reduces exploration capacity, potentially missing better solutions
- High clip fraction (0.2-0.4+) indicates many updates being clipped

**Policy Advantages - SURPRISINGLY STABLE:**
- Maintains near-zero values (~0 to -0.006) which is ideal
- Single spike around step 150 when value function lagged policy changes
- Recovery to near-zero shows healthy PPO dynamics
- Actually the most stable metric in the entire training

#### Root Cause Analysis

1. **Learning Rate Too High**: `1e-5` is too aggressive for the sharp reward landscape created by length-based rewards
2. **Inadequate KL Control**: Despite `init_kl_coef: 1`, the reward signal overwhelms KL penalties
3. **Sharp Reward Landscape**: Length-based rewards create discontinuous optimization surface

#### Conclusion:
While reward optimization is successful, the underlying training dynamics are quite unstable. The training suffers from:
- Failure of KL divergence control
- Severe value function instability  
- Loss of exploration through entropy collapse
- Numerical instability leading to impossible negative KL values

**Training Status**: Successful reward optimization but fundamentally unstable process that could lead to training collapse or unreproducible results.
