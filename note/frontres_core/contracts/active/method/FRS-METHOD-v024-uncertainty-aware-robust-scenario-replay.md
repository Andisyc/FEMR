---
contract_id: FRS-METHOD-v024
status: active
effective_date: 2026-08-12
updated_date: 2026-08-12
supersedes: FRS-METHOD-v023
scope: Bounded outer sealed-Scenario replay with policy-compatible robust expected-utility statistics
---
# Uncertainty-Aware Robust Scenario Replay

## Design delta

FRS-METHOD-v023 ranked Critic calibration replay with the instantaneous
`abs(mean_m U(G_m) - V_old(s))`. Repeated K8 evidence showed that this quantity
tracked current-M4 sampling deviation more strongly than reducible Critic mean
error, so stochastic Scenarios could monopolize replay while still supplying a
noisy Critic target. FRS-METHOD-v024 separates expected utility, rollout
variance, finite-sample uncertainty and policy drift. It changes neither the
Critic's expected-value scientific object nor Actor credit.

## Replay-owned compatible evidence window

For every sealed ScenarioKey and K, Replay owns one committed evidence window:

```text
N_s            = number of compatible committed utility samples
mu_hat_s       = robust estimate of E_pi[U(G) | s]
sigma2_hat_s   = within-Scenario outcome variance
SE_s           = estimated standard error of mu_hat_s
policy_anchor  = 6D diagonal-Gaussian pi_old mean and sigma at window creation
```

The window is built from complete M4 visit batches only and retains at most the
latest 32 compatible visits (128 utility samples). When full, it removes the
oldest complete M4 batch before adding the current one; it never splits a
visit. The robust location is
the 20% symmetric winsorized mean of all samples in the compatible window. It
still estimates a mean: it does not replace expected utility with a median,
maximum, minimum, quantile or risk objective. With fewer than five samples the
trim count is zero, so the first M4 target is the ordinary exact-M mean.

A later visit is compatible only when every row for that Scenario shares one
finite positive diagonal-Gaussian `pi_old` distribution and its symmetric
Gaussian KL to the window anchor is at most `0.02`. This fixed threshold is the
PPO trust-region boundary for replay evidence, not a learned controller. An
incompatible visit starts a new window from its current M4 evidence; it is never
silently pooled through chained pairwise compatibility or transaction age.

## Calibration priority and target

The current visit's shared `V_old(s)` is compared with the candidate window
that includes the current valid M4 batch:

```text
h95_s       = 1.96 * SE_s
E_V_learn_s = max(abs(V_old(s) - mu_hat_s) - h95_s, 0)
target_s    = mu_hat_s
```

`E_V_learn` estimates prediction error beyond plausible sampling noise. High
outcome variance alone is neither a high nor low priority. It widens the mean
uncertainty, while the minimum-four-compatible-visit gate supplies bounded
evidence acquisition without allowing an intrinsically noisy Scenario to block
capacity expansion indefinitely. Capacity maturity is counted only inside the
current policy-compatible window; a policy reset returns that Scenario to one
compatible visit even though its lifetime visit counter remains diagnostic.
`E_A` remains the current-visit centered
absolute utility spread and remains separate from Critic calibration.

Replay stages the candidate records and eight robust targets before PPO. PPO
consumes those targets for Critic loss, while Actor advantage remains the
current attempt's `U(G_sm) - V_old(s)`. Old Actor rows, actions, log
probabilities and advantages are never replayed.

## Bounded selection and transaction

The 64 -> 128 -> 256 active-pool ladder, four DR-class quotas, minimum four
compatible visits, B8 slot schedules, K-local archive and stable ScenarioKey
are unchanged. Calibration slots rank `E_V_learn` plus staleness;
repair-spread slots retain `E_A`; stale review remains explicit. Selection,
evidence-window updates, targets, membership, capacity, staleness and RNG are
one preview. A previously committed transaction identity rejects before a new
plan or stage. The preview commits only after the matching exact-one Adam
receipt.

Checkpoint-v18 stores Replay schema v4, including every record's bounded
compatible window samples, policy anchor, robust mean, outcome variance, standard error,
window reset count, both scores, visits/staleness, active/archive state, RNG and
last commit. Checkpoint-v17 and replay-v3 reject before mutation. Their missing
evidence cannot be synthesized or migrated.

## Preserved boundaries

- Actor remains the deployable 158D full-6D direct Delta SE(3) policy.
- Critic remains the 449D action-pre state value for expected symlog utility.
- Raw GAIN-v008, per-attempt symlog, B8/M4, K8/K16/K32 and DR remain.
- Actor advantage, equal grouped mass, two LR groups, separate clipping and
  exact-one Adam remain.
- Replay never reuses policy rows and never changes Actor loss mass.
- No distributional or variance-head Critic is introduced.

## Falsifiers

- An incompatible policy visit is accumulated into an older window.
- A high-variance, mean-matched Scenario remains high calibration priority only
  because one M4 batch fluctuates.
- A low-variance, wrong-value Scenario loses calibration priority before its
  excess error falls.
- Critic target omits the current M4 batch or Actor advantage uses accumulated
  historical utility.
- A policy-reset Scenario's lifetime visit count opens capacity expansion before
 its new compatible window reaches four visits.
- Failed or duplicate commit changes any replay or window state.
- Checkpoint-v17 is accepted or missing window evidence is zero-filled.
