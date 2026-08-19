---
contract_id: FRS-PPO-v014
status: active-pre-training
effective_date: 2026-08-19
updated_date: 2026-08-19
supersedes: FRS-PPO-v013-for-fresh-relational-training
scope: Actor-only pairwise preference optimization from same-Scenario relational edges
---
# Pairwise Preference Optimization

## 1. Input and ownership

The only supervision is the directed edge set published by FRS-GAIN-v009.
For an edge \((w,l)\in E\), Repair \(w\) is better than Repair \(l\) under
the confirmed hierarchical partial order. `SAME` and `INCOMPARABLE` pairs do
not create edges. `INVALID` fails the transaction closed.

The optimizer owns the full-6D Actor and its trainable action-distribution
parameters only. The compatibility Critic is frozen, excluded from Adam, and
has no target or value loss.

## 2. Loss

For sealed current-policy rows \((s_i,a_i)\), define

\[
d_{wl}(\theta)=\log\pi_\theta(a_w\mid s_w)
                 -\log\pi_\theta(a_l\mid s_l).
\]

The transaction loss is

\[
\mathcal L_{\mathrm{pref}}(\theta)
=\frac{1}{|E|}\sum_{(w,l)\in E}
\operatorname{softplus}\!\left(-\beta d_{wl}(\theta)\right),
\qquad \beta=1.
\]

This route has no old-policy ratio, exponential ratio, clipped surrogate,
advantage, scalar Gain, return, or Critic baseline. The old log-prob remains in
the sealed row only for persistence compatibility and diagnostics; it is not an
input to \(\mathcal L_{\mathrm{pref}}\).

## 3. Learning-rate curriculum

Let \(t\) be the global count of committed relational Actor updates, not the
K-local stage iteration. With \(T_{init}=100\), \(T_{ramp}=50\),
\(\eta_0=3\times10^{-7}\), and \(\eta_1=10^{-6}\):

\[
\eta_t=
\begin{cases}
\eta_0, & 0\le t<T_{init},\\
\eta_0+(\eta_1-\eta_0)\dfrac{t-T_{init}+1}{T_{ramp}},
& T_{init}\le t<T_{init}+T_{ramp},\\
\eta_1, & t\ge T_{init}+T_{ramp}.
\end{cases}
\]

Changing K or lowering DR does not reset \(t\), Adam state, or Actor LR.

## 4. Update and failure boundary

One committed transaction performs exactly one Actor optimizer step. The
existing global Actor gradient-norm safety boundary remains \(0.5\); it is not
a reward, curriculum signal, or substitute for the LR schedule.

If \(|E|=0\), the result is `NO_COMPARABLE_PAIRS` and the transaction is
zero-write. Non-finite current log-probability, malformed edges, or invalid
evidence fails closed before optimizer, Replay, curriculum, or checkpoint
mutation.

## 5. Identity and compatibility

The official objective is

```text
frontres_training_objective=segment_replay_relational_preference_v014
frontres_relational_actor_only=true
frontres_optimization_contract_id=FRS-PPO-v014
```

The launcher selector is `MODE=relational_preference_train`. Checkpoint-v20 and
training telemetry must include `FRS-PPO-v014`,
`pairwise-softplus-logprob-v1`, and `actor-global-100-50-v1`. v013 and v014
checkpoints are not cross-loadable. FRS-PPO-v013 remains a characterized
retired-compatible route only.

Active-pre-training means the selector, loss, LR consumer, telemetry,
checkpoint identity, fake transaction, and persistence boundary are executable
offline. It does not claim simulator execution, convergence, policy quality,
deployment quality, or Formal PASS.
