from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
from typing import Any

import torch

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "frontres_formal_runtime_probe_diagnostics",
    Path(__file__).resolve().with_name("frontres_formal_runtime_probe.py"),
)
assert _AUDIT_SPEC is not None and _AUDIT_SPEC.loader is not None
_AUDIT_MODULE = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(_AUDIT_MODULE)
emit_formal_runtime_probe = _AUDIT_MODULE.emit_formal_runtime_probe


FORBIDDEN_ACCEPTANCE_KEYS = {
    "acceptance_gt",
    "acceptance_mask",
    "acceptance_margin",
    "acceptance_prob",
}


@dataclass(frozen=True)
class FrontRESSegmentReplaySummary:
    scalars: dict[str, float]
    stage: str
    objective: str


_V015_GAIN_SOURCE = "FRS-GAIN-v003-intent-physics-local-repair"
_V015_LOCAL_EVALUATION_KIND = "local_k_candidate_only"
_V015_COMPOSITION_EVALUATION_KIND = "deployment_composition_protocol"


@dataclass(frozen=True)
class FrontRESV015LocalEvaluationReport:
    """Read-only v015 local-K diagnostic projection of sealed candidate evidence.

    Status: active read-only projection for candidate and formal v015 routes.
    Upstream: Step 3B `FrontRESV015GainConsumerEvidence` after its v003 return
    carrier has been sealed.
    Downstream: terminal/diagnostic review only, never sampler, PPO, optimizer,
    checkpoint, or formal evaluator state.
    Evidence: deterministic S1/S2 Physics and transaction contracts.
    Gap: real simulator Physics values remain a bounded live gate.
    """

    transaction_id: str
    scenario_ids: tuple[str, ...]
    noisy_segment_hashes: tuple[str, ...]
    x_t_identities: tuple[str, ...]
    horizon_k: tuple[int, ...]
    policy_actions: tuple[tuple[float, ...], ...]
    valid_policy_row_mask: tuple[bool, ...]
    intent_gain: tuple[float, ...]
    physics_gain: tuple[float, ...]
    repair_cost: tuple[float, ...]
    gain_total: tuple[float, ...]
    policy_values: tuple[float, ...]
    returns: tuple[float, ...]
    raw_advantages: tuple[float, ...]
    repaired_success: tuple[float, ...]
    noisy_success: tuple[float, ...]
    repaired_survival: tuple[float, ...]
    noisy_survival: tuple[float, ...]
    physics_survival_quality_repaired: tuple[float, ...]
    physics_survival_quality_noisy: tuple[float, ...]
    repaired_zmp_margin: tuple[float, ...]
    noisy_zmp_margin: tuple[float, ...]
    repaired_contact: tuple[float, ...]
    noisy_contact: tuple[float, ...]
    physics_success_gain: tuple[float, ...]
    physics_survival_gain: tuple[float, ...]
    physics_zmp_gain: tuple[float, ...]
    physics_contact_gain: tuple[float, ...]
    physics_valid_step_count: tuple[int, ...]
    policy_row_count: int
    valid_policy_row_count: int
    intent_q29_provenance: str
    intent_q29_source: str
    intent_gain_mean: float
    physics_gain_mean: float
    repair_cost_mean: float
    gain_total_mean: float
    gain_total_pos_frac: float
    gain_total_neg_frac: float
    evaluation_kind: str = _V015_LOCAL_EVALUATION_KIND
    gain_source: str = _V015_GAIN_SOURCE
    return_feedback: bool = False
    priority_feedback: bool = False
    ppo_feedback: bool = False

    def validate(self) -> None:
        """Reject non-v003, partial, or feedback-bearing local diagnostic reports."""

        count = int(self.policy_row_count)
        components = (self.intent_gain, self.physics_gain, self.repair_cost, self.gain_total)
        row_diagnostics = (
            self.policy_values,
            self.returns,
            self.raw_advantages,
            self.repaired_success,
            self.noisy_success,
            self.repaired_survival,
            self.noisy_survival,
            self.physics_survival_quality_repaired,
            self.physics_survival_quality_noisy,
            self.repaired_zmp_margin,
            self.noisy_zmp_margin,
            self.repaired_contact,
            self.noisy_contact,
            self.physics_success_gain,
            self.physics_survival_gain,
            self.physics_zmp_gain,
            self.physics_contact_gain,
        )
        if (
            not self.transaction_id
            or count <= 0
            or len(self.scenario_ids) != count
            or len(self.noisy_segment_hashes) != count
            or len(self.x_t_identities) != count
            or len(self.horizon_k) != count
            or len(self.policy_actions) != count
            or any(len(row) != 6 for row in self.policy_actions)
            or len(self.valid_policy_row_mask) != count
            or any(len(values) != count for values in components)
            or any(len(values) != count for values in row_diagnostics)
            or len(self.physics_valid_step_count) != count
            or any(value < 0 or value > self.horizon_k[index] for index, value in enumerate(self.physics_valid_step_count))
            or any(not str(value) for value in self.scenario_ids)
            or any(not str(value) for value in self.noisy_segment_hashes)
            or any(not str(value) for value in self.x_t_identities)
            or any(int(value) <= 0 for value in self.horizon_k)
            or self.valid_policy_row_count < 0
            or self.valid_policy_row_count > count
            or self.valid_policy_row_count != sum(bool(value) for value in self.valid_policy_row_mask)
            or self.evaluation_kind != _V015_LOCAL_EVALUATION_KIND
            or self.gain_source != _V015_GAIN_SOURCE
            or self.intent_q29_provenance != "deployment_noisy_q29"
            or self.return_feedback
            or self.priority_feedback
            or self.ppo_feedback
        ):
            raise ValueError("v015 local evaluation report has invalid identity, Gain source, or feedback boundary")
        if not all(math.isfinite(float(value)) for row in self.policy_actions for value in row):
            raise ValueError("v015 local evaluation report requires finite sealed policy actions [B,6]")
        for row, row_valid in enumerate(self.valid_policy_row_mask):
            row_values = tuple(float(values[row]) for values in (*components, *row_diagnostics))
            if row_valid and not all(math.isfinite(value) for value in row_values):
                raise ValueError("v015 local evaluation report requires finite v003 components on valid policy rows")
            if not row_valid and not all(math.isnan(value) for value in row_values):
                raise ValueError("v015 local evaluation report keeps invalid-row diagnostics UNCONFIRMED, never zero-filled")
        source = self.intent_q29_source.lower()
        if not source or any(token in source for token in ("clean", "root", "global")):
            raise ValueError("v015 local evaluation report rejects non-deployment q29 provenance")
        metrics = (
            self.intent_gain_mean,
            self.physics_gain_mean,
            self.repair_cost_mean,
            self.gain_total_mean,
            self.gain_total_pos_frac,
            self.gain_total_neg_frac,
        )
        if self.valid_policy_row_count > 0:
            if not all(math.isfinite(float(value)) for value in metrics):
                raise ValueError("v015 local evaluation report requires finite v003 diagnostics on valid rows")
            if (
                not 0.0 <= self.gain_total_pos_frac <= 1.0
                or not 0.0 <= self.gain_total_neg_frac <= 1.0
                or self.gain_total_pos_frac + self.gain_total_neg_frac > 1.0 + 1.0e-7
            ):
                raise ValueError("v015 local evaluation report has invalid sign-preserving Gain fractions")
        elif not all(math.isnan(float(value)) for value in metrics):
            raise ValueError("v015 local evaluation report keeps missing diagnostics UNCONFIRMED, never zero-filled")


@dataclass(frozen=True)
class FrontRESV015CompositionEvaluationProtocol:
    """Separate deployment-composition protocol with no local-training feedback channel.

    Status: protocol-only/candidate-only.
    Upstream: an explicitly named deployment reference stream, not local return
    evidence.
    Downstream: a later dedicated sequence evaluator only.
    Evidence: deterministic S1 isolation contract.
    Gap: no sequence simulator execution or composition metric is claimed here.
    """

    reference_stream_id: str
    reference_provenance: str
    frame_count: int
    femr_action_count: int
    evaluation_kind: str = _V015_COMPOSITION_EVALUATION_KIND
    return_feedback: bool = False
    priority_feedback: bool = False
    ppo_feedback: bool = False

    def validate(self) -> None:
        """Reject local-return reuse and invalid deployment-composition protocol facts."""

        if (
            not self.reference_stream_id
            or self.reference_provenance != "deployment_reference_stream"
            or self.frame_count <= 0
            or self.femr_action_count < 0
            or self.femr_action_count > self.frame_count
            or self.evaluation_kind != _V015_COMPOSITION_EVALUATION_KIND
            or self.return_feedback
            or self.priority_feedback
            or self.ppo_feedback
        ):
            raise ValueError("v015 composition protocol has invalid deployment identity or local-training feedback")


def build_frontres_v015_local_evaluation_report(
    candidate_evidence: Any,
    *,
    transaction_id: str,
) -> FrontRESV015LocalEvaluationReport:
    """Project one sealed v003 local candidate carrier into read-only diagnostic facts.

    函数名说明:
        `build_frontres_v015_local_evaluation_report` 是 local-K diagnostic
        projection owner. 它不重算 Gain, 不读取 Clean global metric, 不写 return,
        priority, PPO 或 sampler state.

    主链路:
        上游: Step 3B sealed candidate evidence.
        下游: local evaluator formatter 与 deterministic review contract.

    语义:
        每个 scalar 仅在 valid Repair policy rows 上聚合. 没有 valid row 时保留
        NaN/UNCONFIRMED, 绝不以 0 伪造诊断.
    """

    validate = getattr(candidate_evidence, "validate", None)
    if not callable(validate):
        raise TypeError("v015 local evaluation requires a validated Step 3B candidate carrier")
    validate()
    return_evidence = getattr(candidate_evidence, "return_evidence", None)
    validate_return = getattr(return_evidence, "validate", None)
    if not callable(validate_return):
        raise TypeError("v015 local evaluation requires validated v003 return evidence")
    validate_return()
    if getattr(return_evidence, "gain_source", None) != _V015_GAIN_SOURCE:
        raise ValueError("v015 local evaluation rejects legacy or unspecified Gain source")
    if not str(transaction_id):
        raise ValueError("v015 local evaluation requires sealed transaction identity")

    # B1: 读取 sealed one-row policy metadata, 不读取 mutable sampler state.
    valid_source = getattr(return_evidence, "policy_row_valid", None)
    policy_actions_source = getattr(return_evidence, "policy_actions", None)
    if not isinstance(valid_source, torch.Tensor):
        raise TypeError("v015 local evaluation requires sealed policy_row_valid")
    valid = valid_source.detach().bool().reshape(-1)
    count = int(valid.numel())
    if count <= 0:
        raise ValueError("v015 local evaluation requires at least one policy row")
    if not isinstance(policy_actions_source, torch.Tensor) or tuple(policy_actions_source.shape) != (count, 6):
        raise ValueError("v015 local evaluation requires sealed policy_actions [B,6]")
    components = (
        ("intent_gain", getattr(return_evidence, "intent_gain", None)),
        ("physics_gain", getattr(return_evidence, "physics_gain", None)),
        ("repair_cost", getattr(return_evidence, "repair_cost", None)),
        ("gain_total", getattr(return_evidence, "gain_total", None)),
    )
    if any(not isinstance(value, torch.Tensor) or tuple(value.shape) != (count,) for _, value in components):
        raise ValueError("v015 local evaluation requires row-aligned v003 decomposition tensors")
    diagnostic_names = (
        "policy_values",
        "return_k",
        "advantage_k",
        "repaired_success",
        "noisy_success",
        "repaired_survival",
        "noisy_survival",
        "physics_survival_quality_repaired",
        "physics_survival_quality_noisy",
        "repaired_zmp_margin",
        "noisy_zmp_margin",
        "repaired_contact",
        "noisy_contact",
        "physics_success_gain",
        "physics_survival_gain",
        "physics_zmp_gain",
        "physics_contact_gain",
        "physics_valid_step_count",
    )
    diagnostic_tensors = {name: getattr(return_evidence, name, None) for name in diagnostic_names}
    if any(not isinstance(value, torch.Tensor) or tuple(value.shape) != (count,) for value in diagnostic_tensors.values()):
        raise ValueError("v015 local evaluation requires complete row-aligned Physics/critic diagnostics")

    # B2: 仅在 valid rows 聚合 v003 component, invalid rows 保持 UNCONFIRMED.
    component_mean = {name: _v015_masked_mean(value, valid) for name, value in components}
    gain_total_pos_frac = _v015_masked_sign_fraction(components[3][1], valid, positive=True)
    gain_total_neg_frac = _v015_masked_sign_fraction(components[3][1], valid, positive=False)
    horizon = return_evidence.horizon_k.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if int(horizon.numel()) != count:
        raise ValueError("v015 local evaluation horizon_k must align with policy rows")

    # B3: 构造 immutable report, 明确声明 evaluation 不反馈训练状态.
    report = FrontRESV015LocalEvaluationReport(
        transaction_id=str(transaction_id),
        scenario_ids=tuple(str(value) for value in return_evidence.scenario_ids),
        noisy_segment_hashes=tuple(str(value) for value in return_evidence.noisy_segment_hashes),
        x_t_identities=tuple(str(value) for value in return_evidence.x_t_identities),
        horizon_k=tuple(int(value) for value in horizon.tolist()),
        policy_actions=tuple(
            tuple(float(value) for value in row)
            for row in policy_actions_source.detach().to(device="cpu", dtype=torch.float32).tolist()
        ),
        valid_policy_row_mask=tuple(bool(value) for value in valid.to(device="cpu").tolist()),
        intent_gain=tuple(float(value) for value in components[0][1].detach().to(device="cpu").tolist()),
        physics_gain=tuple(float(value) for value in components[1][1].detach().to(device="cpu").tolist()),
        repair_cost=tuple(float(value) for value in components[2][1].detach().to(device="cpu").tolist()),
        gain_total=tuple(float(value) for value in components[3][1].detach().to(device="cpu").tolist()),
        policy_values=tuple(float(value) for value in diagnostic_tensors["policy_values"].detach().cpu().tolist()),
        returns=tuple(float(value) for value in diagnostic_tensors["return_k"].detach().cpu().tolist()),
        raw_advantages=tuple(float(value) for value in diagnostic_tensors["advantage_k"].detach().cpu().tolist()),
        repaired_success=tuple(float(value) for value in diagnostic_tensors["repaired_success"].detach().cpu().tolist()),
        noisy_success=tuple(float(value) for value in diagnostic_tensors["noisy_success"].detach().cpu().tolist()),
        repaired_survival=tuple(float(value) for value in diagnostic_tensors["repaired_survival"].detach().cpu().tolist()),
        noisy_survival=tuple(float(value) for value in diagnostic_tensors["noisy_survival"].detach().cpu().tolist()),
        physics_survival_quality_repaired=tuple(
            float(value) for value in diagnostic_tensors["physics_survival_quality_repaired"].detach().cpu().tolist()
        ),
        physics_survival_quality_noisy=tuple(
            float(value) for value in diagnostic_tensors["physics_survival_quality_noisy"].detach().cpu().tolist()
        ),
        repaired_zmp_margin=tuple(float(value) for value in diagnostic_tensors["repaired_zmp_margin"].detach().cpu().tolist()),
        noisy_zmp_margin=tuple(float(value) for value in diagnostic_tensors["noisy_zmp_margin"].detach().cpu().tolist()),
        repaired_contact=tuple(float(value) for value in diagnostic_tensors["repaired_contact"].detach().cpu().tolist()),
        noisy_contact=tuple(float(value) for value in diagnostic_tensors["noisy_contact"].detach().cpu().tolist()),
        physics_success_gain=tuple(float(value) for value in diagnostic_tensors["physics_success_gain"].detach().cpu().tolist()),
        physics_survival_gain=tuple(float(value) for value in diagnostic_tensors["physics_survival_gain"].detach().cpu().tolist()),
        physics_zmp_gain=tuple(float(value) for value in diagnostic_tensors["physics_zmp_gain"].detach().cpu().tolist()),
        physics_contact_gain=tuple(float(value) for value in diagnostic_tensors["physics_contact_gain"].detach().cpu().tolist()),
        physics_valid_step_count=tuple(
            int(value) for value in diagnostic_tensors["physics_valid_step_count"].detach().cpu().tolist()
        ),
        policy_row_count=count,
        valid_policy_row_count=int(valid.sum().item()),
        intent_q29_provenance=str(return_evidence.intent_q29_provenance),
        intent_q29_source=str(return_evidence.intent_q29_source),
        intent_gain_mean=component_mean["intent_gain"],
        physics_gain_mean=component_mean["physics_gain"],
        repair_cost_mean=component_mean["repair_cost"],
        gain_total_mean=component_mean["gain_total"],
        gain_total_pos_frac=gain_total_pos_frac,
        gain_total_neg_frac=gain_total_neg_frac,
    )
    report.validate()
    return report


def format_frontres_v015_local_evaluation_report(report: FrontRESV015LocalEvaluationReport) -> str:
    """Format a v015 local-K report without legacy Style or Clean-global fields."""

    report.validate()
    return "\n".join(
        (
            "[FrontRES v015 Local-K Evaluation]",
            (
                "  identity: "
                f"transaction={report.transaction_id} scenarios={report.scenario_ids} "
                f"hashes={report.noisy_segment_hashes} "
                f"x_t={report.x_t_identities} K={report.horizon_k}"
            ),
            (
                "  intent: "
                f"provenance={report.intent_q29_provenance} source={report.intent_q29_source} "
                f"gain={_fmt_v015_eval_scalar(report.intent_gain_mean)}"
            ),
            (
                "  physics: "
                f"gain={_fmt_v015_eval_scalar(report.physics_gain_mean)} "
                f"success_gain={report.physics_success_gain} "
                f"survival=(repair={report.repaired_survival},noisy={report.noisy_survival},"
                f"quality_repair={report.physics_survival_quality_repaired},"
                f"quality_noisy={report.physics_survival_quality_noisy},gain={report.physics_survival_gain}) "
                f"zmp=(repair={report.repaired_zmp_margin},noisy={report.noisy_zmp_margin},gain={report.physics_zmp_gain}) "
                f"contact=(repair={report.repaired_contact},noisy={report.noisy_contact},gain={report.physics_contact_gain})"
            ),
            (
                "  credit: "
                f"value={report.policy_values} return={report.returns} raw_advantage={report.raw_advantages}"
            ),
            f"  repair: cost={_fmt_v015_eval_scalar(report.repair_cost_mean)}",
            (
                "  total: "
                f"gain={_fmt_v015_eval_scalar(report.gain_total_mean)} "
                f"positive={_fmt_v015_eval_percent(report.gain_total_pos_frac)} "
                f"negative={_fmt_v015_eval_percent(report.gain_total_neg_frac)} "
                f"valid_rows={report.valid_policy_row_count}/{report.policy_row_count}"
            ),
            "  boundary: candidate_only=1 return_feedback=0 priority_feedback=0 ppo_feedback=0",
        )
    )


def build_frontres_v015_composition_evaluation_protocol(
    *,
    reference_stream_id: str,
    frame_count: int,
    femr_action_count: int,
    reference_provenance: str = "deployment_reference_stream",
) -> FrontRESV015CompositionEvaluationProtocol:
    """Declare a separate deployment-composition evaluation without local feedback reuse."""

    protocol = FrontRESV015CompositionEvaluationProtocol(
        reference_stream_id=str(reference_stream_id),
        reference_provenance=str(reference_provenance),
        frame_count=int(frame_count),
        femr_action_count=int(femr_action_count),
    )
    protocol.validate()
    return protocol


def format_frontres_v015_composition_evaluation_protocol(
    protocol: FrontRESV015CompositionEvaluationProtocol,
) -> str:
    """Format the protocol boundary; no sequence execution is implied."""

    protocol.validate()
    return "\n".join(
        (
            "[FrontRES v015 Deployment Composition Protocol]",
            f"  reference: id={protocol.reference_stream_id} provenance={protocol.reference_provenance}",
            f"  execution: frames={protocol.frame_count} femr_actions={protocol.femr_action_count}",
            "  boundary: local_return_feedback=0 replay_priority_feedback=0 ppo_feedback=0",
        )
    )


def _v015_masked_mean(value: torch.Tensor, valid: torch.Tensor) -> float:
    data = value.detach().float().reshape(-1)
    if tuple(data.shape) != tuple(valid.shape):
        raise ValueError("v015 diagnostic component mask must align with values")
    selected = data[valid]
    if selected.numel() == 0:
        return float("nan")
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("v015 diagnostic component is nonfinite on a valid policy row")
    return float(selected.mean().cpu().item())


def _v015_masked_sign_fraction(value: torch.Tensor, valid: torch.Tensor, *, positive: bool) -> float:
    data = value.detach().float().reshape(-1)
    if tuple(data.shape) != tuple(valid.shape):
        raise ValueError("v015 diagnostic positivity mask must align with values")
    selected = data[valid]
    if selected.numel() == 0:
        return float("nan")
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("v015 diagnostic total Gain is nonfinite on a valid policy row")
    sign_mask = selected > 0.0 if positive else selected < 0.0
    return float(sign_mask.float().mean().cpu().item())


def _fmt_v015_eval_scalar(value: float) -> str:
    return f"{float(value):.6f}" if math.isfinite(float(value)) else "UNCONFIRMED"


def _fmt_v015_eval_percent(value: float) -> str:
    return f"{float(value) * 100.0:.1f}%" if math.isfinite(float(value)) else "UNCONFIRMED"


def summarize_segment_batch(
    sample: Any,
    reward_result: Any,
    reset_result: Any,
    action_stats: Any,
    sampler_stats: Any | None = None,
    stage: str = "stage3_segment_hrl",
    objective: str = "segment_replay_hrl",
) -> FrontRESSegmentReplaySummary:
    scalars: dict[str, float] = {}
    sources = tuple(getattr(sample, "source", ()))
    total = max(1, len(sources))
    scalars["segment/global_frac"] = sources.count("global") / total
    scalars["segment/replay_frac"] = sources.count("replay") / total
    scalars["segment/review_frac"] = sources.count("review") / total
    priority = getattr(sample, "priority", torch.zeros(0))
    scalars["segment/priority_mean"] = _mean(priority)
    scalars["segment/priority_p90"] = _quantile(priority, 0.9)
    if sampler_stats is not None:
        scalars["segment/replay_pool_size"] = float(getattr(sampler_stats, "replay_pool_size", 0))
    else:
        scalars["segment/replay_pool_size"] = float((priority > 0.0).sum().item()) if isinstance(priority, torch.Tensor) else 0.0

    solved = getattr(reward_result, "solved_mask", torch.zeros(0, dtype=torch.bool))
    hopeless = getattr(reward_result, "hopeless_mask", torch.zeros_like(solved))
    valid = getattr(reward_result, "valid_mask", torch.ones_like(solved))
    scalars["segment/solved_frac"] = _bool_mean(solved)
    scalars["segment/hopeless_frac"] = _bool_mean(hopeless)
    scalars["segment/active_frac"] = _bool_mean(valid & (~solved.bool()) & (~hopeless.bool())) if isinstance(valid, torch.Tensor) else 0.0
    scalars["segment/reset_success_frac"] = _bool_mean(getattr(reset_result, "success_mask", torch.zeros(0, dtype=torch.bool)))
    scalars["segment/preroll_frac"] = _bool_mean(getattr(reset_result, "preroll_mask", torch.zeros(0, dtype=torch.bool)))
    horizon = getattr(sample, "horizon_k", None)
    if horizon is None:
        horizon = getattr(reward_result, "horizon_k", None)
    scalars["segment/k"] = _mean(horizon) if isinstance(horizon, torch.Tensor) else float(horizon or 0.0)
    scalars["segment/score_noisy"] = _mean(getattr(reward_result, "score_noisy", torch.zeros(0)))
    scalars["segment/score_repaired"] = _mean(getattr(reward_result, "score_repaired", torch.zeros(0)))
    scalars["segment/score_clean"] = _mean(getattr(reward_result, "score_clean", torch.zeros(0)))
    scalars["segment/gain_over_noisy"] = _mean(getattr(reward_result, "gain_over_noisy", torch.zeros(0)))
    scalars["segment/fall_frac"] = _bool_mean(getattr(reward_result, "fall_flag", torch.zeros(0, dtype=torch.bool)))
    scalars["segment/contact_consistency"] = _mean(getattr(reward_result, "contact_consistency", torch.zeros(0)))
    scalars["segment/action_norm"] = float(getattr(action_stats, "action_norm_mean", 0.0))
    per_dim = getattr(action_stats, "per_dim_norm", torch.zeros(6))
    per_dim = per_dim.detach().flatten().float().cpu() if isinstance(per_dim, torch.Tensor) else torch.zeros(6)
    labels = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")
    for i, label in enumerate(labels):
        scalars[f"segment/action_norm_{label}"] = float(per_dim[i].item()) if i < per_dim.numel() else 0.0
    # B2: Remove retired acceptance fields before terminal/logger formatting.
    for key in FORBIDDEN_ACCEPTANCE_KEYS:
        scalars.pop(key, None)
    return FrontRESSegmentReplaySummary(scalars=scalars, stage=stage, objective=objective)


def format_segment_replay_log(summary: FrontRESSegmentReplaySummary) -> str:
    scalars = summary.scalars
    return (
        f"FrontRES Segment HRL active: stage={summary.stage} objective={summary.objective} "
        f"k={scalars.get('segment/k', 0.0):.0f} "
        f"mix=global:{scalars.get('segment/global_frac', 0.0):.2f}/"
        f"replay:{scalars.get('segment/replay_frac', 0.0):.2f}/"
        f"review:{scalars.get('segment/review_frac', 0.0):.2f} "
        f"gain={scalars.get('segment/gain_over_noisy', 0.0):.4f} "
        f"reset={scalars.get('segment/reset_success_frac', 0.0):.2f}"
    )


def repair_effect_summary_to_scalars(summary: dict[str, Any]) -> dict[str, float]:
    """把 canonical Gain summary 转换为 train-effect scalars.

    函数名说明:
        `repair_effect_summary_to_scalars` 是 diagnostic projection owner, 只选择和
        命名正式 Gain/sampler 字段; 它不是 Gain 公式, 也不回写训练状态.

    主链路:
        上游: live probe 和 update-loop aggregation 产生 canonical summary.
        下游: terminal/logger formatter 消费扁平 scalars.

    语义:
        诊断必须读取 Style/Physics/Repair Cost/Total Gain 的正式 owner 字段.
        缺失 motion/physics evidence 保持 UNCONFIRMED, legacy acceptance score 被忽略.
    """
    # B1: 从 final live summary 读取 canonical Gain 和 sampler fields.
    scalars = {
        "segment/train_effect_gain_style": _optional_float(summary.get("gain_style_mean")),
        "segment/train_effect_gain_physics": _optional_float(summary.get("gain_physics_mean")),
        "segment/train_effect_repair_cost": _optional_float(summary.get("gain_repair_cost_mean")),
        "segment/train_effect_gain_total": _optional_float(summary.get("gain_total_mean")),
        "segment/train_effect_gain_pos_frac": _optional_float(summary.get("gain_total_pos_frac")),
        "segment/train_effect_fall_rate": _optional_float(summary.get("done_frac")),
        "segment/train_effect_valid_frac": _optional_float(summary.get("storage_valid_frac")),
        "segment/train_effect_replay_candidates": _summary_float(
            summary,
            "sampler_update_replay_candidate_count",
            "sampler_replay_candidates",
        ),
        "segment/train_effect_replay_pool_size": _float(summary.get("sampler_replay_pool_size")),
    }
    for key in FORBIDDEN_ACCEPTANCE_KEYS:
        scalars.pop(key, None)
    # B3: AUDIT-DIAG-01 截获 terminal/logger formatter 实际消费的 scalars 和 transaction 聚合状态.
    # Result: PENDING_LIVE.
    emit_formal_runtime_probe(
        "AUDIT-DIAG-01",
        scalars=scalars,
        audit_identity_mode=summary.get("audit_identity_mode", "UNCONFIRMED"),
        audit_transaction_count=summary.get("audit_transaction_count", 0),
        audit_transaction_ids=summary.get("audit_transaction_ids", ()),
        audit_batch_signature_count=summary.get("audit_batch_signature_count", 0),
        audit_batch_signatures=summary.get("audit_batch_signatures", ()),
        audit_same_transaction=summary.get("audit_same_transaction", False),
    )
    return scalars


def format_segment_train_effect_log(summary: dict[str, Any]) -> str:
    scalars = repair_effect_summary_to_scalars(summary)
    return "\n".join(
        (
            "[FrontRES Segment Train Effect]",
            (
                "  audit: "
                f"mode={summary.get('audit_identity_mode', 'UNCONFIRMED')} "
                f"transactions={summary.get('audit_transaction_count', 0)} "
                f"batches={summary.get('audit_batch_signature_count', 0)} "
                f"same_transaction={summary.get('audit_same_transaction', False)} "
                f"transaction_ids={summary.get('audit_transaction_ids', ())} "
                f"batch_signatures={summary.get('audit_batch_signatures', ())}"
            ),
            (
                "  gain: "
                f"style={_fmt_motion_scalar(scalars, 'segment/train_effect_gain_style')} "
                f"physics={_fmt_motion_scalar(scalars, 'segment/train_effect_gain_physics')} "
                f"repair_cost={_fmt_motion_scalar(scalars, 'segment/train_effect_repair_cost')} "
                f"total={_fmt_motion_scalar(scalars, 'segment/train_effect_gain_total')} "
                f"gain_pos={_fmt_motion_percent(scalars, 'segment/train_effect_gain_pos_frac')}"
            ),
            (
                "  data: "
                f"fall={_fmt_motion_percent(scalars, 'segment/train_effect_fall_rate')} "
                f"valid={_fmt_motion_percent(scalars, 'segment/train_effect_valid_frac')}"
            ),
            (
                "  replay: "
                f"candidates={scalars['segment/train_effect_replay_candidates']:.0f} "
                f"pool={scalars['segment/train_effect_replay_pool_size']:.0f}"
            ),
        )
    )


def motion_quality_summary_to_scalars(
    *,
    clean_positions: torch.Tensor | None = None,
    repaired_positions: torch.Tensor | None = None,
    noisy_positions: torch.Tensor | None = None,
    delta_se: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    temporal_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    valid = valid_mask.bool() if isinstance(valid_mask, torch.Tensor) else None
    temporal = temporal_mask.bool() if isinstance(temporal_mask, torch.Tensor) else None
    return {
        "segment/motion_mpjpe_repaired_clean": _mpjpe(repaired_positions, clean_positions, valid, temporal),
        "segment/motion_mpjpe_noisy_clean": _mpjpe(noisy_positions, clean_positions, valid, temporal),
        "segment/motion_vel_error_repaired_clean": _diff_mpjpe(repaired_positions, clean_positions, valid, temporal, order=1),
        "segment/motion_acc_error_repaired_clean": _diff_mpjpe(repaired_positions, clean_positions, valid, temporal, order=2),
        "segment/motion_delta_se_norm": _delta_se_norm(delta_se, None),
        "segment/motion_delta_z_up_frac": _delta_z_up_frac(delta_se, None),
    }


def format_segment_motion_quality_log(scalars: dict[str, float]) -> str:
    return "\n".join(
        (
            "[FrontRES Segment Motion Quality]",
            (
                "  pose: "
                f"mpjpe_repaired={_fmt_motion_scalar(scalars, 'segment/motion_mpjpe_repaired_clean')} "
                f"mpjpe_noisy={_fmt_motion_scalar(scalars, 'segment/motion_mpjpe_noisy_clean')}"
            ),
            (
                "  dynamics: "
                f"vel_err={_fmt_motion_scalar(scalars, 'segment/motion_vel_error_repaired_clean')} "
                f"acc_err={_fmt_motion_scalar(scalars, 'segment/motion_acc_error_repaired_clean')}"
            ),
            (
                "  action: "
                f"delta_se_norm={_fmt_motion_scalar(scalars, 'segment/motion_delta_se_norm')} "
                f"dz_up={_fmt_motion_percent(scalars, 'segment/motion_delta_z_up_frac')}"
            ),
        )
    )


def periodic_eval_summary_to_scalars(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "segment/eval_episode_length": _float(summary.get("episode_length")),
        "segment/eval_success_rate": _float(summary.get("success_rate")),
        "segment/eval_fall_rate": _float(summary.get("fall_rate")),
        "segment/eval_mean_survival_steps": _float(summary.get("mean_survival_steps")),
        "segment/eval_gain_style": _optional_float(summary.get("gain_style_mean")),
        "segment/eval_gain_physics": _optional_float(summary.get("gain_physics_mean")),
        "segment/eval_gain_repair_cost": _optional_float(summary.get("gain_repair_cost_mean")),
        "segment/eval_gain_total": _optional_float(summary.get("gain_total_mean")),
        "segment/eval_gain_total_pos_frac": _optional_float(summary.get("gain_total_pos_frac")),
        "segment/eval_motion_mpjpe_repaired_clean": _optional_float(summary.get("segment/motion_mpjpe_repaired_clean")),
        "segment/eval_motion_mpjpe_noisy_clean": _optional_float(summary.get("segment/motion_mpjpe_noisy_clean")),
        "segment/eval_motion_vel_error_repaired_clean": _optional_float(summary.get("segment/motion_vel_error_repaired_clean")),
        "segment/eval_motion_acc_error_repaired_clean": _optional_float(summary.get("segment/motion_acc_error_repaired_clean")),
        "segment/eval_motion_delta_se_norm": _optional_float(summary.get("segment/motion_delta_se_norm")),
        "segment/eval_motion_delta_z_up_frac": _optional_float(summary.get("segment/motion_delta_z_up_frac")),
    }


def action_distribution_health_summary(
    *,
    means: torch.Tensor | None = None,
    sigmas: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    supervised_target: torch.Tensor | None = None,
    raw_saturation_warn: float = 2.0,
    raw_saturation_bad: float = 20.0,
) -> dict[str, float | str | bool]:
    """Summarize whether a task-space policy distribution is numerically usable."""
    summary: dict[str, float | str | bool] = {
        "available": isinstance(means, torch.Tensor) and means.numel() > 0,
        "status": "UNCONFIRMED",
        "raw_mean_abs_max": _unconfirmed(),
        "raw_mean_abs_mean": _unconfirmed(),
        "raw_saturated_frac_abs_gt_2": _unconfirmed(),
        "raw_saturated_frac_abs_gt_20": _unconfirmed(),
        "sigma_min": _unconfirmed(),
        "sigma_mean": _unconfirmed(),
        "sigma_max": _unconfirmed(),
        "action_norm_mean": _unconfirmed(),
        "target_norm_mean": _unconfirmed(),
        "action_norm_over_target_norm": _unconfirmed(),
    }
    if not isinstance(means, torch.Tensor) or means.numel() == 0:
        return summary

    raw = means.detach().float()
    finite = torch.isfinite(raw)
    if not bool(finite.all().item()):
        summary["status"] = "BAD_NONFINITE_RAW_MEAN"
        return summary

    abs_raw = raw.abs()
    raw_abs_max = float(abs_raw.max().cpu().item())
    raw_abs_mean = float(abs_raw.mean().cpu().item())
    sat_warn = float((abs_raw > float(raw_saturation_warn)).float().mean().cpu().item())
    sat_bad = float((abs_raw > float(raw_saturation_bad)).float().mean().cpu().item())
    summary.update(
        {
            "raw_mean_abs_max": raw_abs_max,
            "raw_mean_abs_mean": raw_abs_mean,
            "raw_saturated_frac_abs_gt_2": sat_warn,
            "raw_saturated_frac_abs_gt_20": sat_bad,
        }
    )

    if isinstance(sigmas, torch.Tensor) and sigmas.numel() > 0:
        sigma = sigmas.detach().float()
        if not bool(torch.isfinite(sigma).all().item()):
            summary["status"] = "BAD_NONFINITE_SIGMA"
            return summary
        summary.update(
            {
                "sigma_min": float(sigma.min().cpu().item()),
                "sigma_mean": float(sigma.mean().cpu().item()),
                "sigma_max": float(sigma.max().cpu().item()),
            }
        )

    action_norm = _row_norm_mean(actions)
    target_norm = _row_norm_mean(supervised_target)
    summary["action_norm_mean"] = action_norm
    summary["target_norm_mean"] = target_norm
    if math.isfinite(action_norm) and math.isfinite(target_norm) and target_norm > 1e-8:
        summary["action_norm_over_target_norm"] = action_norm / target_norm

    if raw_abs_max >= float(raw_saturation_bad) or sat_bad > 0.0:
        summary["status"] = "BAD_RAW_MEAN_SATURATED"
    elif sat_warn >= 0.50:
        summary["status"] = "WARN_RAW_MEAN_SATURATING"
    else:
        summary["status"] = "OK"
    return summary


def format_segment_periodic_eval_log(summary: dict[str, Any]) -> str:
    scalars = periodic_eval_summary_to_scalars(summary)
    families = dict(summary.get("perturbation_family_counts", {}) or {})
    return "\n".join(
        (
            "[FrontRES Segment Periodic Eval]",
            (
                "  batch: "
                f"source={summary.get('eval_batch_source', 'UNCONFIRMED')} "
                f"reset={bool(summary.get('eval_reset_applied', False))} "
                f"motion_ids={tuple(summary.get('motion_ids', ()) or ())} "
                f"start_frames={tuple(summary.get('start_frames', ()) or ())}"
            ),
            (
                "  perturbation: "
                f"families={families} "
                f"strength_min={float(summary.get('perturbation_strength_min', 0.0)):.6f} "
                f"strength_mean={float(summary.get('perturbation_strength_mean', 0.0)):.6f} "
                f"strength_max={float(summary.get('perturbation_strength_max', 0.0)):.6f}"
            ),
            (
                "  rollout: "
                f"episode_length={scalars['segment/eval_episode_length']:.1f} "
                f"survival={scalars['segment/eval_mean_survival_steps']:.1f}"
            ),
            (
                "  result: "
                f"success={scalars['segment/eval_success_rate'] * 100.0:.1f}% "
                f"fall={scalars['segment/eval_fall_rate'] * 100.0:.1f}%"
            ),
            (
                "  gain: "
                f"source={summary.get('gain_source', 'UNCONFIRMED')} "
                f"style={_fmt_eval_scalar(scalars, 'segment/eval_gain_style')} "
                f"physics={_fmt_eval_scalar(scalars, 'segment/eval_gain_physics')} "
                f"repair_cost={_fmt_eval_scalar(scalars, 'segment/eval_gain_repair_cost')} "
                f"total={_fmt_eval_scalar(scalars, 'segment/eval_gain_total')} "
                f"positive={_fmt_eval_percent(scalars, 'segment/eval_gain_total_pos_frac')}"
            ),
            (
                "  motion: "
                f"mpjpe_repaired={_fmt_eval_scalar(scalars, 'segment/eval_motion_mpjpe_repaired_clean')} "
                f"mpjpe_noisy={_fmt_eval_scalar(scalars, 'segment/eval_motion_mpjpe_noisy_clean')} "
                f"vel_err={_fmt_eval_scalar(scalars, 'segment/eval_motion_vel_error_repaired_clean')} "
                f"acc_err={_fmt_eval_scalar(scalars, 'segment/eval_motion_acc_error_repaired_clean')}"
            ),
            (
                "  action: "
                f"delta_se_norm={_fmt_eval_scalar(scalars, 'segment/eval_motion_delta_se_norm')} "
                f"dz_up={_fmt_eval_percent(scalars, 'segment/eval_motion_delta_z_up_frac')}"
            ),
        )
    )


def segment_summary_to_scalars(summary: FrontRESSegmentReplaySummary) -> dict[str, float]:
    scalars = dict(summary.scalars)
    for key in FORBIDDEN_ACCEPTANCE_KEYS:
        scalars.pop(key, None)
    return scalars


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().float().mean().cpu().item())
    if value is None:
        return 0.0
    return float(value)


def _optional_float(value: Any) -> float:
    if value is None:
        return _unconfirmed()
    return _float(value)


def _summary_float(summary: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return _float(value)
    return 0.0


def _unconfirmed() -> float:
    return float("nan")


def _fmt_motion_scalar(scalars: dict[str, float], key: str) -> str:
    value = float(scalars.get(key, _unconfirmed()))
    if not math.isfinite(value):
        return "UNCONFIRMED"
    return f"{value:.6f}"


def _fmt_motion_percent(scalars: dict[str, float], key: str) -> str:
    value = float(scalars.get(key, _unconfirmed()))
    if not math.isfinite(value):
        return "UNCONFIRMED"
    return f"{value * 100.0:.1f}%"


def _fmt_eval_scalar(scalars: dict[str, float], key: str) -> str:
    return _fmt_motion_scalar(scalars, key)


def _fmt_eval_percent(scalars: dict[str, float], key: str) -> str:
    return _fmt_motion_percent(scalars, key)


def _mpjpe(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    valid: torch.Tensor | None,
    temporal: torch.Tensor | None,
) -> float:
    if not _same_position_shape(a, b):
        return _unconfirmed()
    diff = torch.linalg.norm(a.float() - b.float(), dim=-1)
    return _masked_batch_mean(diff, valid, temporal)


def _diff_mpjpe(
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    valid: torch.Tensor | None,
    temporal: torch.Tensor | None,
    *,
    order: int,
) -> float:
    if not _same_position_shape(a, b) or a.shape[1] <= order:
        return _unconfirmed()
    da = torch.diff(a.float(), n=order, dim=1)
    db = torch.diff(b.float(), n=order, dim=1)
    diff_temporal = temporal[:, order:] if temporal is not None and temporal.ndim == 2 else None
    return _masked_batch_mean(torch.linalg.norm(da - db, dim=-1), valid, diff_temporal)


def _same_position_shape(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
    return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape and a.ndim >= 3


def _masked_batch_mean(
    value: torch.Tensor,
    valid: torch.Tensor | None,
    temporal: torch.Tensor | None = None,
) -> float:
    if value.numel() == 0:
        return 0.0
    if valid is not None and valid.shape[0] == value.shape[0]:
        value = value[valid]
        if temporal is not None and temporal.shape[0] == valid.shape[0]:
            temporal = temporal[valid]
        if value.numel() == 0:
            return 0.0
    if temporal is not None and value.ndim >= 2 and tuple(temporal.shape) == tuple(value.shape[:2]):
        value = value[temporal]
        if value.numel() == 0:
            return 0.0
    return float(value.mean().item())


def _delta_se_norm(delta_se: torch.Tensor | None, valid: torch.Tensor | None) -> float:
    if not isinstance(delta_se, torch.Tensor) or delta_se.numel() == 0:
        return 0.0
    value = torch.linalg.norm(delta_se.float(), dim=-1)
    if valid is not None and valid.shape[0] == value.shape[0]:
        value = value[valid]
        if value.numel() == 0:
            return 0.0
    return float(value.mean().item())


def _delta_z_up_frac(delta_se: torch.Tensor | None, valid: torch.Tensor | None) -> float:
    if not isinstance(delta_se, torch.Tensor) or delta_se.ndim < 2 or delta_se.shape[-1] < 3:
        return 0.0
    value = delta_se[..., 2] > 0.0
    if valid is not None and valid.shape[0] == value.shape[0]:
        value = value[valid]
        if value.numel() == 0:
            return 0.0
    return float(value.float().mean().item())


def _row_norm_mean(value: torch.Tensor | None) -> float:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return _unconfirmed()
    data = value.detach().float()
    if data.ndim == 1:
        data = data.view(1, -1)
    if not bool(torch.isfinite(data).all().item()):
        return _unconfirmed()
    return float(torch.linalg.norm(data, dim=-1).mean().cpu().item())


def _mean(value: torch.Tensor | None) -> float:
    if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(value.float().mean().item())


def _quantile(value: torch.Tensor | None, q: float) -> float:
    if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(torch.quantile(value.float().flatten(), q).item())


def _bool_mean(value: torch.Tensor | None) -> float:
    if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
        return 0.0
    return float(value.bool().float().mean().item())
