### train/kl -- KL divergence to reference
Average KL(policy || reference), typically per token. Measures how far you’re drifting.

Small but non-zero. Many setups aim around 0.01–0.1 per step/batch (depends on seq length, scaling).

### train/policy_loss -- policy gradient term
The clipped surrogate objective. Because we maximize reward, many libs log it as a negative number (more negative ≈ better improvement).

### train/value_loss -- critic regression loss
MSE loss between predicted value and target return/advantage. Downward trend early; then small oscillations.

### policy/entropy -- exploration
Mean token-level entropy of the policy distribution.
Starts higher, decreases slowly as model becomes confident.

### policy/approx_kl -- surrogate KL
A fast proxy derived from log-prob ratios during PPO training. Used as a sanity check inside PPO updates. Can differ in scale/sign from true KL, but should trend similarly.

### policy/clip-fraction -- % of samples that were clipped
Fraction where |r_t − 1| exceeded ε (PPO clipping).

### policy/advantages_mean — mean advantage
Batch mean of advantages (should be centered).

### value/prediction_error — critic error

### value/explained_variance — R²-like signal
how good is my value function at predicting returns
Near 1 = great; 0 = no better than mean; <0 = worse than constant. If it’s negative → critic is diverging.

### value/clip_fraction — value clipping rate
Fraction of value updates clipped (PPO value clipping).
