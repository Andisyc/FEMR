---
contract_id: FRS-PPO-v013
status: active-pre-training
effective_date: 2026-08-17
updated_date: 2026-08-17
supersedes: FRS-PPO-v012-for-relational-route
scope: Clipped Actor PPO from same-Scenario preference edges; no value loss
---
# Relational Actor-Only PPO

The input is a complete sealed transaction with policy rows and directed
preference edges. Edges are the only learning label. No scalar Gain, return,
mean target, value normalizer, or Critic advantage is constructed.

For edge incidence \(c_i=W_i-L_i\), the clipped loss is:

\[
L_{pair}= -\frac{1}{|E|}\sum_i
\min\big(r_i c_i,\operatorname{clip}(r_i,1-\epsilon,1+\epsilon)c_i\big),
\qquad r_i=\exp(\log\pi_\theta(a_i|s_i)-\log\pi_{old}(a_i|s_i)).
\]

Only Actor parameters (and any explicitly trainable policy distribution
parameter) belong to the optimizer. Compatibility Critic parameters are
frozen and excluded. The optimizer must have exactly one `frontres_role=actor`
group. No edge means zero-write: no backward, optimizer step, Replay commit,
curriculum advance, or checkpoint write.

The legacy scalar route remains characterized under FRS-PPO-v012 and cannot
consume relational batches.
