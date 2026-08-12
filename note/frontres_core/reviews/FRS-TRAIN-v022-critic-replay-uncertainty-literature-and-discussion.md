---
record_id: FRS-TRAIN-v022-critic-replay-uncertainty-review
status: evidence-only
recorded_date: 2026-08-12
checkout: aa3c4f786a3c61ef377a7e9777f0a1f2aaa0d062
method_contract: FRS-METHOD-v023
training_contract: FRS-TRAIN-v022
optimization_contract: FRS-PPO-v010
scope: Literature evidence and design discussion about stochastic Scenario targets, robust mean estimation, uncertainty and Replay priority
---

# Critic Replay Uncertainty: Literature And Discussion Record

## Authority and evidence boundary

This is an evidence-only research and discussion record. It does not activate a
new method, change the Design Inspector, authorize implementation, alter the
checkpoint schema, or recommend starting another training run.

The current active route remains:

```text
FRS-METHOD-v023
FRS-TRAIN-v022
FRS-PPO-v010
frontres-v022-checkpoint-v17
```

The current 449D Critic remains the action-pre state value `V(s)`. The active
target is the current-Transaction M4 mean of per-attempt symmetric-log utility:

```text
U(G_m) = sign(G_m) * log1p(abs(G_m))
value_target_s = mean_m U(G_m)
advantage_sm = U(G_sm) - V_old(s)
```

Any change from the current M4 target to a cross-visit estimate is a semantic
change and therefore requires a confirmed Design Inspector, coordinated
Contract versions, persistence review and new runtime evidence.

## Question recovered from the discussion

The discussion began from three observations or hypotheses:

1. one unusually large Repair Gain may still pull a four-sample utility mean;
2. repeated fresh-policy M4 visits to the same sealed Scenario may produce a
   target distribution whose within-Scenario variation is larger than the
   stable difference between Scenario means;
3. ranking Replay by a large instantaneous Critic calibration error can then
   repeatedly select stochastic Scenarios rather than Scenarios on which the
   Critic has reducible error.

The second item is a working diagnosis inherited from the preceding log
discussion. Its exact numerical evidence must be revalidated against the named
local log/report in the main session before it is promoted to a runtime fact.

## Current implementation boundary

`FrontRESOuterScenarioReplay.stage()` currently derives two values from each
Scenario's M4 Actor advantages:

```text
E_V = abs(mean_m advantage_m)
    = abs(mean_m U(G_m) - V_old(s))

E_A = mean_m abs(advantage_m - mean_m advantage_m)
    = mean_m abs(U(G_m) - mean_m U(G_m))
```

`FrontRESScenarioReplayRecord.with_visit()` applies a fixed EMA independently
to these scores. The record persists visit count and score EMA, but it does not
own a cross-visit estimator of the Scenario utility mean, outcome variance,
standard error, confidence interval, or reducible-versus-irreducible
uncertainty.

Consequently, a high `E_V` currently has two indistinguishable explanations:

- the Critic mean is wrong and can be improved by learning;
- this visit's M4 mean is an unusually noisy realization even though the
  Critic is close to the Scenario's stable expected utility.

The second case can remain high priority without providing learnable Critic
information. This is the precise boundary investigated by the literature
search below.

## Literature findings

### 1. Error-prioritized Replay can chase irreducible noise

[Uncertainty Prioritized Experience Replay](https://arxiv.org/abs/2506.09270)
is the closest direct match. It distinguishes:

- **aleatoric uncertainty**: variation caused by the outcome distribution,
  which cannot be removed merely by fitting the value function longer;
- **epistemic uncertainty**: uncertainty about the expected value that can be
  reduced by learning.

The paper argues that TD-error priority can favor unpredictable transitions
even when the value estimator is already near the target mean. It replaces raw
error priority with an information-gain criterion using both epistemic and
aleatoric uncertainty. Candidate forms include ratios or normalized quantities
such as `E/A` and `E^2/U`; the exact form is model- and data-dependent.

FEMR implication: raw `E_V` is total observed discrepancy, not automatically
reducible Critic error. Outcome dispersion must participate in Replay's notion
of learnability.

### 2. Mean and outcome dispersion are different prediction objects

[A Distributional Perspective on Reinforcement Learning](https://proceedings.mlr.press/v70/bellemare17a.html)
models the return distribution while retaining its expectation as the ordinary
value. [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/abs/1804.08617)
shows that a distributional Critic is compatible with continuous-control
Actor-Critic: the Actor can still use the expected value while the Critic
represents more than one moment.

[Normality-Guided Distributional Reinforcement Learning for Continuous Control](https://arxiv.org/abs/2208.13125)
provides a closer PPO precedent. It separates the mean value estimate from a
variance network and evaluates the method on continuous-control PPO tasks.

FEMR implication: retaining an expected-utility Critic while separately
estimating outcome dispersion is a mature design family. Variance does not
replace the mean and does not have to enter the Actor objective.

### 3. Extreme or heavy-tailed outcomes require robust mean estimation

[No-Regret Reinforcement Learning with Heavy-Tailed Rewards](https://proceedings.mlr.press/v130/zhuang21a.html)
and [Robust Offline Reinforcement Learning with Heavy-Tailed Rewards](https://proceedings.mlr.press/v238/zhu24a.html)
show that ordinary sample means and value estimates can be fragile under
heavy-tailed rewards. The latter uses median-of-means to retain a mean-estimation
objective while reducing domination by rare extreme observations.

FEMR already applies symmetric-log utility before the M4 mean. This compresses
raw Gain magnitude but does not remove sign flips, multimodality, or large
within-Scenario outcome dispersion. A robust estimator across repeated visits
can therefore remain relevant even after symlog.

### 4. Increasing the rollout horizon is not a variance remedy

[Averaging n-step Returns Reduces Variance](https://proceedings.mlr.press/v235/daley24a.html)
analyzes how longer multi-step returns can increase variance and studies
compound averaging as a variance-reduction mechanism.

FEMR implication: increasing `K` changes the physical consequence horizon and
may increase target variability. It is not a substitute for estimating the
same-Scenario mean or uncertainty. Increasing `M` lowers Monte Carlo standard
error only at approximately `1/sqrt(M)` and still does not tell Replay whether
an error is reducible.

## Distilled design conclusions

### Confirmed conceptual conclusions

1. The Critic's scientific object remains expected utility. Variance, maximum
   or minimum should not silently replace that object.
2. A large observed calibration error is not sufficient evidence that a
   Scenario has high learning value.
3. Variance is necessary to interpret error, but raw variance alone is not a
   complete Replay priority.
4. A useful confidence estimate needs sample count. One M4 batch is too small
   to characterize a Scenario reliably; statistics should accumulate across
   committed fresh-policy visits to the same stable Scenario identity.
5. Outcome variance must remain separate from uncertainty that can be reduced
   by fitting the Critic.
6. Variance must not enter Actor advantage or make the method risk-sensitive
   unless that objective change is separately proposed and confirmed.

### Minimal candidate semantic delta

Replay, rather than Actor, should own per-Scenario cross-visit statistics:

```text
N_s                 committed valid utility samples or visit batches
mu_hat_s            robust estimate of expected U(G) under the relevant policy window
sigma2_hat_s        estimated within-Scenario outcome variance
SE_s                estimated uncertainty of the mean, approximately sigma_hat_s / sqrt(N_s)
```

Replay priority should represent error beyond plausible sampling noise, not
variance itself. One interpretable candidate is:

```text
excess_error_s = max(abs(V_old(s) - mu_hat_s) - c * SE_s, 0)
```

This formula is only an explanatory candidate, not an activated design. A
final priority must also retain bounded coverage/staleness and define how
policy drift limits the lifetime of accumulated statistics.

The intended ordering is:

- large prediction error with low outcome uncertainty: high Replay value;
- large instantaneous error with high irreducible variance: do not let it
  dominate Replay indefinitely;
- too few observations: collect enough evidence before declaring the Critic
  wrong;
- stable mean already matched by the Critic: low calibration priority even if
  individual outcomes remain diverse.

## Options and semantic cost

### Option A: uncertainty-aware Replay ranking only

Keep the active current-Transaction M4 Critic target and Actor advantage. Add
cross-visit Replay statistics and use them only for future Scenario selection.

- smallest optimization change;
- can prevent noisy Scenarios from monopolizing Replay;
- does not remove noise from the M4 target used by each Critic update;
- should be described as mitigation, not guaranteed closure.

### Option B: ranking plus robust cross-visit Critic target

Use accumulated current-policy-compatible evidence to supervise the Scenario
mean as well as to rank Replay.

- directly addresses both selection pollution and target pollution;
- requires an explicit policy-window/staleness rule because Actor updates
  change the target distribution;
- changes the active exact-M target semantics, checkpoint payload and formal
  transaction contract;
- requires a new Design Inspector decision and coordinated Method/Training/PPO
  versions.

### Option C: distributional or variance-head Critic

Predict mean plus variance/quantiles and optionally estimate epistemic
uncertainty through ensembles or another explicit estimator.

- has mature research precedent;
- can generalize uncertainty to Scenarios with few visits;
- adds network, loss, checkpoint and calibration complexity;
- is not the first minimal change while Replay already has exact stable
  Scenario identities and repeated visits.

## Recommended decision order

1. Revalidate the repeated-Scenario log evidence locally and distinguish
   within-Scenario target variance from between-Scenario mean separation.
2. Confirm the scientific object: Critic predicts expected symlog utility;
   uncertainty is Replay evidence, not a new Actor objective.
3. Decide explicitly between Option A and Option B. Do not describe Option A as
   a complete target-noise repair.
4. Define the policy-compatibility window before accumulating cross-visit
   statistics. Old-policy outcomes cannot be pooled indefinitely as if the
   target distribution were stationary.
5. Only if replay-owned statistics cannot generalize or remain too sparse,
   consider Option C.

## Required falsification evidence before another long run

- repeated sealed-Scenario visits expose `N`, robust mean, variance and standard
  error without changing Scenario identity;
- one injected extreme utility cannot materially dominate the accumulated mean
  or Replay rank;
- a high-variance Scenario with a stable matched mean does not remain the top
  calibration priority solely because of fresh M4 fluctuations;
- a low-variance Scenario with a genuinely wrong value remains high priority
  until the prediction error falls;
- Replay coverage, staleness, DR compatibility, B8/M4 grouped mass, exact-one
  update and Actor advantage semantics remain unchanged;
- a bounded K8 test demonstrates improved Critic separation/calibration before
  any K transition or long campaign.

## Governance receipt

```text
classification: evidence-only research and proposal input
active semantic authority changed: no
Design Inspector changed: no
production code changed: no
training authorized: no
next safe action: main-session evidence revalidation and human choice between
                  ranking-only mitigation and ranking-plus-target repair
```
