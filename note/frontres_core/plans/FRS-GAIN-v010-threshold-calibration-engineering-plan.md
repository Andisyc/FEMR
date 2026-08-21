# FRS-GAIN-v010 Threshold Calibration Engineering Plan

Status: proposal / pre-training only / no active-route change

## Objective

建立一份可复核的阈值与噪声容差校准 artifact，使 FRS-GAIN-v010 的层级比较能够区分真实物理变化与测量波动。该计划不把 Clean 均值当作安全真值，不引入标量总分，也不改变 active `FRS-GAIN-v009`。

## Preserved behavior and forbidden scope

- 保留 `FRS-GAIN-v009` 的层级安全优先、同层 Pareto、`INCOMPARABLE` 和 `INVALID` 语义。
- 保留物理失败不可补偿；噪声容差不能覆盖真实失载、非计划换脚或非法接触。
- 不修改 active Gain、PPO/Loss、Actor、Critic、Replay、训练配置、GMT 或部署路径。
- 不用训练曲线、edge 数量、L3 比例或策略质量选择阈值。
- 不在线更新阈值，不为缺失数据 zero-fill，不把 N/A 误写为安全。

## Measurement objects

每个连续字段 `j` 必须绑定单位、方向、时间窗口和有效性条件：

- Capture Margin 及其趋势；
- 适用支撑相位内的 ZMP；
- 线动量和角动量误差；
- 支撑脚漂移；
- 稳定保持时间。

严重事件必须保存独立标签：survival、expected/actual load、planned/actual support transition、illegal contact。

## Bounded execution steps

### S0 — Freeze the calibration card

冻结字段顺序、单位、坐标方向、时间窗、采样频率、`N/A` 条件、`alpha` 候选值和独立事件标签规则。

Stop if any field has no semantic owner or its applicability cannot be stated.

### S1 — Repeated no-change baseline

在已知稳定状态下重复相同 Scenario、动作和传感器流程，分别检查：

1. 传感器重复读数；
2. 估计器对相同原始数据的重复输出；
3. 允许仿真重复时的接触/数值波动。

对每个窗口级指标计算：

\[
 D_j^{(a,b)}=\phi_j(C^{(a)})-\phi_j(C^{(b)}),\qquad
 \delta_j=Q_{1-\alpha}(|D_j|).
\]

Artifact 必须记录单位、样本数、覆盖率、时间窗、Scenario/GMT/cache identity 和 hash。单次 Clean 不能生成该 artifact。

Stop if `delta_j` is non-finite, zero without a justified exact measurement, identity mismatches, or repeated identical states are not mostly `SAME`.

### S2 — Controlled perturbation sweep

固定 GMT、机器人、初始状态和参考动作，只改变一种扰动因素。使用相对强度扫描，例如 `0, 0.25, 0.5, 0.75, 1.0, 1.25` 倍基础扰动，每个等级重复执行，并记录完整逐步轨迹和独立事件标签。

扫描的目标是找到安全区间和失败边界，不是最大化 L3 或比较训练 checkpoint。

Stop if increasing perturbation produces a safety improvement without an actual recovery intervention, or if event labels cannot distinguish failure from measurement noise.

### S3 — Threshold construction

- Survival、失载、非计划换脚、非法接触：使用独立物理事件定义；
- Capture/ZMP：使用可行域符号零点，零点附近再应用 `delta_j`；
- 连续 Recovery：使用 S1 得到的 `delta_j` 作为分辨率；
- Stable hold：由任务终止语义定义 `W`，不能从训练曲线拟合；
- Applicability：由接触相位、承重和有效时间窗决定。

输出安全区间而不是单点：`[safe_lower, safe_upper]`。若不存在满足约束的区间，保留 `INCONCLUSIVE`，不继续扩大容差。

### S4 — Comparator pseudo-samples

准备入口：`source/rsl_rl/rsl_rl/tests/frontres_gain_threshold_calibration_alignment.py`。它调用 Clean-relative public producer 与 consumer；在真实重复测量数据尚未存在时，不得把该入口描述为完整 runtime calibration。

通过独立比较器边界构造并验证：

- 相同稳定状态的差异落在噪声容差内，结果为 `SAME`；
- 单一指标超过容差且其余指标不变，结果为 `BETTER/WORSE`；
- 两个指标分别改善和恶化，结果为 `INCOMPARABLE`；
- 严重物理失败不能被普通 Recovery 或 Intent 补偿；
- 缺失、非法、适用性不一致返回 `INVALID`。

Stop if tolerance makes clearly different states indistinguishable, or makes identical states directional.

### S5 — Held-out validation

使用未参与校准的 Scenario、扰动方向和强度区间重复 S2–S4。报告：

- hard-event false-safe 数量；
- hard-event false-unsafe 数量；
- 单调性反转数量；
- SAME/INCOMPARABLE 比例；
- 每个连续字段的有效覆盖率。

不得通过改变阈值直到 held-out 通过；失败时返回 `OBJECTIVE-VIOLATION`、`INCONCLUSIVE` 或 `TELEMETRY-GAP`。

### S6 — Freeze and admission review

只有在 S0–S5 全部通过后，才冻结 calibration artifact，并进行 `MODULE-CORRECT`、code review 和 formal-route review。artifact 进入 active route 仍需单独的 Design Inspector/Contract 确认；本计划本身不授权训练或 live test。

### S7 — Read-only repeated-Clean collection adapter (current engineering unit)

The adapter boundary is deliberately typed and output-oriented:

```text
CleanCalibrationCollectionRequest
  + ReadOnlyCleanCollection
  -> ReadOnlyCleanCollectionReceipt
  -> immutable CleanCalibration
```

The request binds the exact Segment/cache/Clean artifact, expected-support,
GMT checkpoint/normalizer, K, timestep, field schema and repeat seed protocol.
The collection result must contain one complete window per requested repeat,
the same identity echoed by every window, closed-repeat IDs, collector identity,
and unchanged training/RNG-restore hashes before and after the whole campaign
and each repeat. Each repeat seed is hashed separately; it may differ between
repeats, while the restore hash must return to the campaign boundary. Missing
rows, identity drift, partial cleanup, training mutation or RNG drift fail
closed before a calibration artifact is emitted.

This unit only validates a completed gateway result and builds the existing
calibration artifact. It does not implement simulator reset/materialization,
does not accept a generic callback, and does not enter the active v009 Gain or
TRAIN-v025 route. The official composition-root gateway remains a separate
formal-runtime boundary.

### S7a — Official route map (pre-construction gate)

The future official offline gateway must use this single route and no legacy
evaluation owner:

```text
scripts/rsl_rl/train.py
  --frontres_stage stage3_segment_hrl
  --frontres_clean_calibration_collect_only
  -> OnPolicyRunner composition root
  -> run_frontres_clean_calibration_collect()
  -> frontres_readonly_collection_scope(route="clean_calibration")
  -> existing Stage-1 cache/reset/materialization owners
  -> run_frontres_clean_calibration_collect_typed(request, prepared)
  -> repeated Clean telemetry producer
  -> typed CleanCalibrationCollectionRequest
  -> ReadOnlyCleanWindow / ReadOnlyCleanCollection
  -> adapt_read_only_clean_collection()
  -> ReadOnlyCleanCollectionReceipt
```

Route identity is `FRS-EVAL-v010-clean-calibration-v001` and the only new CLI
selector is `--frontres_clean_calibration_collect_only`. Its effective branch
must set `evaluation_only=true`, `frontres_segment_replay_enabled=false`,
`frontres_policy_quality_eval_only=false`, `frontres_action_gain_direction_collect_only=false`,
`frontres_local_sentinel_only=false`, and `frontres_segment_update_steps=0`.
The branch must require the typed request/manifest and result path, and reject
any optimizer, Replay, legacy policy-quality, or action-Gain operation before
runner construction.

The route must reject `policy_quality_eval`, action-Gain collection, training
Replay, optimizer updates, active `FRS-GAIN-v009`, and the legacy v006 Gain
consumer. The final semantic consumer is the pure
`adapt_read_only_clean_collection()` boundary; no scalar Gain or training
consumer is permitted. The external simulator/cache operation is the only
replaceable seam in the official offline pseudo-transaction.

The producer uses `collect_frontres_v017_no_actor_baseline` for every
requested repeat, with `clean_baseline` reset, sealed Scenario/K identity and
distinct repeat seeds restored at the campaign boundary. Continuous fields
are reduced by the existing relational Outcome owner; hard-event labels are
retained as typed `CleanHardEventEvidence`. Any hard event, identity drift,
state/RNG mutation or incomplete repeat aborts before a calibration artifact
is emitted.

## Acceptance card

`MODULE-CORRECT` requires:

1. known same-state repeats are mostly `SAME`;
2. supra-resolution one-sided changes are ordered;
3. real cross-metric conflicts remain `INCOMPARABLE`;
4. hard Physics violations remain lexicographically dominant;
5. missing/invalid/applicability failures are `INVALID`;
6. held-out perturbations preserve monotonicity;
7. artifact identity, units, coverage and hash are verified.

## Current stop boundary

The deterministic producer/consumer and semantic pseudo-samples are now
module-correct offline. The typed adapter and raw producer are connected to
the official CLI manifest branch through the version-neutral fixed-K/M4
materializer and the existing read-only scope. The connector constructs a
typed request from the sealed Stage-1 owner, rejects incomplete identity or
normalizer provenance, and emits no artifact on preparation failure. This is
still not R1 evidence: formal-runtime admission must prove one unchanged
composition-root pseudo-transaction reaches the adapter through the external
simulator/cache seam, then separately establish real telemetry. No active Gain
integration or training launch is authorized by this document.
