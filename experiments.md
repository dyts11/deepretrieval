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

### experiment 2
#### training description
Reduced learning rate from 1e-5 to 1e-6 while keeping all other hyperparameters identical to experiment 1.
Testing if more conservative learning rate can achieve stable training.

#### training results
![](graph/train_2_train.png)
![](graph/train_2_policy.png)

#### Detailed Analysis

**Training Stability - DRAMATICALLY IMPROVED:**
- **Value Loss**: Extremely stable throughout training, staying consistently low (~0.5-3.0 range)
- **Policy Loss**: Much more stable, oscillating gently around 0-0.015 range  
- **Total Loss**: Dominated by stable value loss, showing consistent training dynamics
- **No catastrophic spikes**: Complete elimination of the massive instabilities from experiment 1

**KL Divergence Control - MIXED RESULTS:**
- **Much more controlled**: KL values stay in manageable -3 to +1 range
- **Persistent negative KL**: Most updates show KL between 0 to -3, which is still concerning
- **No extreme spikes**: Eliminated the dangerous 20-300 KL spikes from experiment 1
- **Better but not perfect**: Still indicates some numerical/computational issues

**Value Function Performance - EXCELLENT:**
- **Prediction Error**: Stable and low throughout training (1-4 range vs 14+ in experiment 1)
- **Explained Variance**: Steady improvement from -6 to 0, indicating value function learning well
- **Clip Fraction**: Very low and stable, showing appropriate update magnitudes

**Policy Behavior - STABLE BUT LIMITED LEARNING:**
- **Entropy**: More gradual decline from ~1.8 to ~2.5, maintaining some exploration
- **Clip Fraction**: Very low (~0.01), indicating small, conservative policy updates
- **Approx KL**: Consistently low (~0.0005), showing minimal policy changes

**Reward Learning - CONCERNING LACK OF PROGRESS:**
- **Minimal improvement**: Reward stays mostly flat around 0.9-1.2 range
- **No clear learning trend**: Unlike experiment 1's clear upward trajectory
- **Model barely adapting**: Learning rate may now be too conservative

#### Stability vs Learning Trade-off Analysis

**What We Gained:**
✅ **Eliminated catastrophic instabilities** (value loss spikes, extreme KL divergence)  
✅ **Stable value function** learning and prediction  
✅ **Controlled policy updates** without dangerous divergence  
✅ **Reproducible training** without numerical issues  
✅ **Better entropy management** maintaining exploration longer  

**What We Lost:**
❌ **Minimal reward optimization** - model not learning the task effectively  
❌ **Slow adaptation** to reward signals  
❌ **Potential under-exploration** of policy space  
❌ **Learning efficiency** significantly reduced  

#### Root Cause Analysis

1. **Over-Conservative Learning Rate**: `1e-6` may be too small for meaningful policy updates
2. **Reward Signal Insufficient**: Weak gradients not strong enough to drive learning
3. **KL Penalty Dominance**: KL control now overpowering reward optimization  
4. **Exploration-Exploitation Balance**: Too much emphasis on stability, insufficient exploration

#### Negative KL Persistence

The persistent negative KL (-3 to 0) suggests:
- **Computational precision issues**: Even with stable training, some numerical problems remain
- **Reference model drift**: Possible issues with reference model consistency  
- **Log probability calculations**: Still some instability in probability computations
- **Not immediately dangerous**: Unlike experiment 1, these are manageable magnitudes

#### Conclusion:
Experiment 2 successfully solved the stability issues but at the cost of learning effectiveness. The 10x learning rate reduction created a **stable but under-performing training regime**. 

**Key Finding**: Need to find the optimal learning rate between 1e-6 (too conservative) and 1e-5 (too aggressive).

**Training Status**: Stable training achieved but insufficient task learning - requires learning rate tuning.
