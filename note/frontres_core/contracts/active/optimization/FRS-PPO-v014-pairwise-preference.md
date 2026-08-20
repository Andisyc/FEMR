---
contract_id: FRS-PPO-v014
status: active-pre-training
effective_date: 2026-08-19
updated_date: 2026-08-20
supersedes: FRS-PPO-v013-for-fresh-relational-training
scope: Actor-only pairwise preference optimization from same-Scenario relational edges
---
# Pairwise Preference Optimization

## 1. Input and ownership

The only supervision is the directed edge set published by FRS-GAIN-v009.
For an edge \((w,l)\in E\), Repair \(w\) is better than Repair \(l\) under
the confirmed hierarchical partial order. `SAME` and `INCOMPARABLE` pairs do
not create edges. `INVALID` fails the transaction closed.

The optimizer owns the full-6D Actor mean network only. Task-space sigma is a
positive isotropic buffer, excluded from Adam; a trainable sigma is invalid for
this contract. The compatibility Critic is frozen, excluded from Adam, and
has no target or value loss.

## 2. Loss

Each sealed row also stores the transaction-start reference distribution:
\(\mu_i^{ref},\sigma_i^{ref},\log\pi_{ref}(a_i\mid s_i)\). The current
Actor is evaluated on the same \((s_i,a_i)\). For the task-space Actor, each
six-dimensional Gaussian must use one positive isotropic standard deviation
\(\sigma_i\). Before the Loss is called, the formal transaction must verify
that the current Actor state hash still equals the sealed policy snapshot; this
identity check, rather than a second floating-point forward comparison, rejects
a stale reference. The current and sealed sigma must match exactly within the
declared numeric tolerance.

A repeated CUDA forward of the same Actor snapshot may differ by small
floating-point roundoff, including when the same rows are evaluated under a
different batch shape.
The score therefore uses a reference-valued, current-gradient mean

\[
\bar\mu_i(\theta)=\mu_i^{ref}+
\left(\mu_i(\theta)-\operatorname{stopgrad}(\mu_i(\theta))\right),
\]

whose forward value is exactly \(\mu_i^{ref}\) and whose derivative is the
current Actor derivative. Define the reference-relative Fisher-scaled row score

\[
r_i(\theta)=(\sigma_i^{ref})^2\left[
\log\mathcal N(a_i;\bar\mu_i(\theta),\sigma_i^{ref})
-\log\pi_{ref}(a_i\mid s_i)
\right].
\]

For an edge \((w,l)\), define

\[
d_{wl}(\theta)=r_w(\theta)-r_l(\theta).
\]

The transaction first averages edge losses within each Scenario and then
averages Scenario losses:

\[
\mathcal L_{pref}(\theta)=
\frac{1}{|\mathcal S|}\sum_{S\in\mathcal S}
\frac{1}{|E_S|}\sum_{(w,l)\in E_S}
\operatorname{softplus}\!\left(-d_{wl}(\theta)\right).
\]

The reference KL is reported as a drift diagnostic:

\[
\mathcal L_{KL}(\theta)=
\frac{1}{|I_E|}\sum_{i\in I_E}\frac12\sum_{d=1}^{6}
\left[
\frac{\sigma_{i,d}^2+(\mu_{i,d}-\mu^{ref}_{i,d})^2}{(\sigma^{ref}_{i,d})^2}
-1+2\log\frac{\sigma^{ref}_{i,d}}{\sigma_{i,d}}
\right].
\]

Here \(I_E\) is the set of valid rows referenced by at least one preference
edge; uninvolved rows do not contribute even to this diagnostic.

The active objective is

\[
\mathcal L(\theta)=\mathcal L_{pref}(\theta).
\]

The row factor \(\sigma_i^2\) removes the diagonal Gaussian mean-Fisher
scale for this isotropic task-space distribution. It is a row-wise
approximation, not an exact full-network natural gradient. The historical
direct log-prob preference function remains only as a characterization
baseline; it is not consumed by the active formal route.

The formal transaction evaluates this objective once before one optimizer
step. The current Actor is the sealed transaction reference at that point, so
the diagnostic KL is zero by construction and is not a one-step trust-region
solver. Any future multi-step or post-step KL constraint requires a separately
versioned update contract.

The active route has no old-policy ratio, exponential ratio, clipped surrogate,
advantage, scalar Gain, return, or Critic baseline.

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

The formal transaction owner is the only v014 update seam; the legacy direct
live-policy update seam rejects v014. One committed transaction performs
exactly one Actor optimizer step. The
existing global Actor gradient-norm safety boundary remains \(0.5\); it is not
a reward, curriculum signal, or substitute for the LR schedule.

If \(|E|=0\), the result is `NO_COMPARABLE_PAIRS` and the transaction is
zero-write. A stale policy-snapshot hash, trainable or non-isotropic sigma,
current sigma/reference-sigma mismatch, current or sealed Gaussian/log-prob
inconsistency, any invalid row, non-finite policy value, malformed edge, or
invalid evidence fails closed before optimizer, Replay, curriculum, or
checkpoint mutation. A mean-only floating-point recomputation difference does
not override a matching policy-snapshot hash.

## 5. Identity and compatibility

The official objective is

```text
frontres_training_objective=segment_replay_relational_preference_v014
frontres_relational_actor_only=true
frontres_optimization_contract_id=FRS-PPO-v014
```

The launcher selector is `MODE=relational_preference_train`. Checkpoint-v20 and
training telemetry must include `FRS-PPO-v014`,
`pairwise-reference-fisher-scenario-v1`, and `actor-global-100-50-v1`. v013 and v014
checkpoints are not cross-loadable. FRS-PPO-v013 remains a characterized
retired-compatible route only.

Active-pre-training means the selector, loss, LR consumer, telemetry,
checkpoint identity, fake transaction, and persistence boundary are executable
offline. It does not claim simulator execution, convergence, policy quality,
deployment quality, or Formal PASS.
