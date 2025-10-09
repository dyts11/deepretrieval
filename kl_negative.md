### What kl means in TRL PPO
kl = (logprobs - ref_logprobs).mean() 
Measured KL (output log) → “How far has the actor drifted?
KL coefficient (penalty strength) → “How hard do we pull it back?”
TRL warns because sustained negative KL often means:
Your generation kwargs mismatch
Policy is collapsing toward trivial completions where logprobs > ref.



objective/kl
This is the “controlled KL” term used in the PPO objective.
It measures how far the policy distribution has moved from the reference model distribution.

In TRL, it’s usually computed as:
KL(𝜋𝜃∣∣ 𝜋ref)=𝐸𝑎∼𝜋𝜃[log⁡𝜋𝜃(𝑎∣𝑠)−log⁡𝜋ref(𝑎∣𝑠)]

This is the value actually penalized by the kl_coef.
It should generally stay positive and relatively small. If it grows too large, the policy is drifting away too much.

Expected behavior:
Starts near 0, then hovers in a small, stable range (often 0.01–0.2 depending on scaling).
If it explodes → policy updates are too aggressive.
If it collapses to ~0 → the KL penalty is too strong (policy is barely learning).

ppo/policy/approxkl
This is the approximate KL used as a diagnostic in PPO.
Notice the difference:

objective/kl: new policy vs. reference model

approxkl: new policy vs. old policy (the one before the PPO update step)
That’s why approx_kl can go negative → if the new policy happens to assign higher probability to sampled actions than the old one, the mean log-ratio becomes negative.
PPO uses this for early stopping (if KL > target, stop updating in this epoch).

Expected behavior:
Fluctuates around 0.
Should remain relatively small (e.g. |approx_kl| < 0.01–0.05 per update).
Large swings → unstable updates.


