# FrontRES Engineering Plan

Status update, 2026-06-27:

The acceptance-only HSL+HRL engineering plan below is no longer the preferred
next research direction.  The current concept is recorded in
`note/FrontRES Segment Replay HRL Method Design.md`:

- slice long motions into dynamic segments;
- use HSL as initialization, not as the final proposal to accept/reject;
- train HRL to output full 6D Delta SE repair with segment-level PPO;
- coordinate global coverage and repeated rollout through prioritized segment
  replay.

The older acceptance plan remains useful as historical implementation context
until a new engineering plan replaces it.

This document is the engineering plan for turning the active FEMR design
contract into code.  It is not a history log.  When Dr. Cheng says "write the
code modification plan into ./note", update this file unless the request is
only about research concept or checklist status.

## 1. Active Engineering Goal

Implement the simplified FEMR HSL+HRL acceptance design:

```text
corrupted reference
  -> Stage 1 HSL Clean-oriented Delta SE proposal
  -> Stage 2 HRL / acceptance decision
  -> Delta SE_exec = accept * Delta SE_HSL
  -> frozen GMT execution
```

This replaces the active Authority Actor-Critic path.  The older authority
critic, continuous rho actor, structured-rho advantage, and alpha-rho branches
are ablation/history only unless a future plan explicitly revives them.

## 2. Fixed Design Decisions

```text
Stage 1:
  owns continuous Delta SE proposal magnitude and direction.
  trained by supervised/HSL evidence from clean/noisy reference pairs.

Stage 2:
  owns admissibility of the proposal.
  receives observation + detached Stage-1 proposal.
  trained from Noisy-vs-Candidate rollout comparison.

Execution:
  Delta SE_exec = accept * Delta SE_HSL.
  accept may be binary or a probability used as near-binary confidence.

Forbidden active path:
  no authority critic Q(s, d, rho);
  no actor update by maximizing Q;
  no continuous rho advantage as main objective;
  no alpha/rho legacy payload in the active training loss.
```

## 3. Step Plan

| Step | Goal | Main Modules | Required Test |
| --- | --- | --- | --- |
| 1 | Rewrite method contract and checklist | `note/*.md` | `frontres_design_contract_sentinel.py` |
| 2 | Config cleanup | `rsl_rl_mosaic_cfg.py`, `rsl_rl_cfg.py` | `frontres_config_hsl_acceptance.py` |
| 3 | Policy surface cleanup | `front_residual_actor_critic.py` | `frontres_hsl_acceptance_policy.py` |
| 4 | Rollout evidence and labels | `frontres_rollout_step.py`, `frontres_post_step_connector.py`, `frontres_rollout_evidence.py` | `frontres_acceptance_label_from_rollout.py` |
| 5 | Storage contract | `rollout_storage.py` | `frontres_acceptance_storage_contract.py` |
| 6 | Algorithm loss | `frontres_unified.py` | `frontres_acceptance_algorithm_loss.py` |
| 7 | Runner live path cleanup | `on_policy_runner.py`, `frontres_training_setup.py` | `frontres_runner_hsl_acceptance_path.py` |
| 8 | Diagnostics cleanup | `frontres_diagnostics.py`, `frontres_reward_diagnostics.py`, `frontres_runner_logging.py` | `frontres_hsl_acceptance_diagnostics.py` |
| 9 | Entrypoint scripts | `run/run_frontres_stage1_hsl.sh`, `run/run_frontres_stage2_acceptance.sh` | `frontres_stage_entrypoint_contract.py` |
| 10 | Full toy chain | tests only | `frontres_hsl_acceptance_full_chain.py` |
| 11 | Legacy active-path audit | all active files | `frontres_no_legacy_active_path.py` |
| 12 | Training readiness suite | tests/checklist | `frontres_hsl_acceptance_training_readiness.py` |

## 4. Acceptance Label Contract

Stage 2 must be trained by a direct rollout comparison:

```text
margin = Candidate_exec_score - Noisy_exec_score

if margin > positive_margin:
  accept_gt = 1
  accept_mask = 1
elif margin < negative_margin:
  accept_gt = 0
  accept_mask = 1
else:
  accept_mask = 0 or low weight
```

The label says whether the Stage-1 proposal should be written.  It does not ask
Stage 2 to discover the exact continuous write strength.

## 5. Required Diagnostics

The active log should expose:

```text
proposal magnitude
acceptance probability / rate
accept_gt rate
accept_mask rate
Candidate-Noisy margin
accepted-beneficial fraction
accepted-harmful fraction
rejected-harmful fraction
authority actor-critic disabled sentinel
```

Old authority diagnostics such as `authority Q`, `authority actor ready`,
`rho near0/near1`, and `authority critic loss` must not appear in the active
FEMR log except under an explicit ablation name.

## 6. Test Discipline

Every step must add or update a cheap test before recommending a training run.
The test should prove the local contract and, when possible, the route into the
next module.  A short live run is allowed only after the active path can be
proven by static tests and diagnostic sentinels.
