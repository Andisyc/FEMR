from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class FrontRESSegmentReplayStepResult:
    sample: Any
    batch: Any
    reset_request: Any
    reset_result: Any
    raw_action: Any
    policy_output: Any
    repair_action: Any
    rollout: Any
    reward_result: Any
    priority_evidence: Any
    diagnostics: Any
    log_string: str


class FrontRESSegmentReplayConnector:
    """Thin connector for Segment Replay HRL.

    This object owns call order only.  Dataset, sampler, reset, action
    projection, rollout, reward, and diagnostics keep their own logic.
    """

    def __init__(
        self,
        *,
        dataset: Any,
        sampler: Any,
        reset_adapter: Any,
        action_projector: Any,
        reward: Any,
        rollout_fn: Callable[..., Any],
        transition_writer: Any | None = None,
        diagnostics_fn: Callable[..., Any] | None = None,
        log_formatter: Callable[[Any], str] | None = None,
        reset_mode: str = "auto",
        stage: str = "stage3_segment_hrl",
        objective: str = "segment_replay_hrl",
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.reset_adapter = reset_adapter
        self.action_projector = action_projector
        self.reward = reward
        self.rollout_fn = rollout_fn
        self.transition_writer = transition_writer
        self.diagnostics_fn = diagnostics_fn
        self.log_formatter = log_formatter
        self.reset_mode = reset_mode
        self.stage = stage
        self.objective = objective

    def run_step(self, *, env: Any, policy: Any, batch_size: int, command: Any | None = None) -> FrontRESSegmentReplayStepResult:
        sample = self.sampler.sample(batch_size)
        batch = self.dataset.get_segments(sample.segment_ids)
        reset_request = self.reset_adapter.build_request(batch, mode=self.reset_mode)
        reset_result = self.reset_adapter.apply(env, reset_request)
        policy_output = self._policy_action(policy, batch, reset_result)
        raw_action = self._policy_action_tensor(policy_output)
        mode_groups = self._mode_groups(batch)
        repair_action = self.action_projector.project(raw_action, mode_groups=mode_groups)
        command = self.action_projector.apply_to_reference({} if command is None else command, repair_action)
        rollout = self.rollout_fn(
            env=env,
            batch=batch,
            reset_request=reset_request,
            reset_result=reset_result,
            repair_action=repair_action,
            command=command,
        )
        reward_result = self.reward.compute(
            self._rollout_role(rollout, "noisy"),
            self._rollout_role(rollout, "repaired"),
            self._rollout_role(rollout, "clean"),
            reset_result,
        )
        priority_evidence = self.reward.priority_evidence(
            reward_result,
            batch.segment_ids,
            getattr(batch, "horizon_k", 0),
        )
        self.sampler.update(priority_evidence)
        self._write_transition(
            sample=sample,
            batch=batch,
            reset_request=reset_request,
            reset_result=reset_result,
            raw_action=raw_action,
            policy_output=policy_output,
            repair_action=repair_action,
            rollout=rollout,
            reward_result=reward_result,
            priority_evidence=priority_evidence,
        )
        diagnostics = self._diagnostics(sample, reward_result, reset_result, repair_action)
        log_string = self.log_formatter(diagnostics) if self.log_formatter is not None and diagnostics is not None else ""
        return FrontRESSegmentReplayStepResult(
            sample=sample,
            batch=batch,
            reset_request=reset_request,
            reset_result=reset_result,
            raw_action=raw_action,
            policy_output=policy_output,
            repair_action=repair_action,
            rollout=rollout,
            reward_result=reward_result,
            priority_evidence=priority_evidence,
            diagnostics=diagnostics,
            log_string=log_string,
        )

    def _policy_action(self, policy: Any, batch: Any, reset_result: Any) -> Any:
        if hasattr(policy, "act_segment"):
            return policy.act_segment(batch=batch, reset_result=reset_result)
        if hasattr(policy, "act"):
            return policy.act(batch)
        if callable(policy):
            return policy(batch)
        raise TypeError("policy must define act_segment, act, or be callable")

    def _policy_action_tensor(self, policy_output: Any) -> Any:
        if isinstance(policy_output, dict):
            for key in ("action", "actions", "raw_action"):
                if key in policy_output:
                    return policy_output[key]
        for key in ("action", "actions", "raw_action"):
            value = getattr(policy_output, key, None)
            if value is not None:
                return value
        return policy_output

    def _mode_groups(self, batch: Any) -> tuple[tuple[str, ...], ...] | None:
        families = getattr(batch, "perturbation_family", None)
        if families is None:
            return None
        return tuple((family,) if isinstance(family, str) else tuple(family) for family in families)

    def _rollout_role(self, rollout: Any, role: str) -> Any:
        if isinstance(rollout, dict):
            return rollout[role]
        return getattr(rollout, role)

    def _write_transition(self, **payload: Any) -> None:
        if self.transition_writer is None:
            return
        if hasattr(self.transition_writer, "write"):
            self.transition_writer.write(**payload)
            return
        self.transition_writer(**payload)

    def _diagnostics(self, sample: Any, reward_result: Any, reset_result: Any, repair_action: Any) -> Any:
        if self.diagnostics_fn is None:
            return None
        action_stats = self.action_projector.stats(repair_action)
        sampler_stats = self.sampler.stats() if hasattr(self.sampler, "stats") else None
        return self.diagnostics_fn(
            sample,
            reward_result,
            reset_result,
            action_stats,
            sampler_stats=sampler_stats,
            stage=self.stage,
            objective=self.objective,
        )
