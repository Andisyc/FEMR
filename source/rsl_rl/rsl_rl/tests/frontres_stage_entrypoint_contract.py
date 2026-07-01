#!/usr/bin/env python3
"""FrontRES stage entrypoint contract.

Stage 1/2 are live.  Stage 3 is recognized as Segment Replay HRL.  Explicit
sentinel/probe/storage/single-update/update-loop sentinels and the dedicated
live train loop can enter the minimal runner path.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def _between(text: str, start: str, end: str) -> str:
    start_i = text.index(start)
    end_i = text.index(end, start_i)
    return text[start_i:end_i]


def main() -> None:
    train = _read("scripts/rsl_rl/train.py")
    stage1_cache = _read("run/run_frontres_stage1_segment_cache.sh")
    stage1_cache_validator = _read("run/validate_frontres_stage1_segment_cache.sh")
    stage1 = _read("run/run_frontres_stage1_hsl.sh")
    stage2 = _read("run/run_frontres_stage2_acceptance.sh")
    stage3 = _read("run/run_frontres_stage3_segment_hrl.sh")
    root_stage1 = _read("run_stage1.sh")
    root_stage2 = _read("run_stage2.sh")
    root_stage3 = _read("run_stage3.sh")

    assert 'choices=("stage1_segment_cache", "stage1_hsl", "stage2_hsl_warmup", "stage2_acceptance", "stage3_segment_hrl")' in train
    assert '"--frontres_segment_cache_dir"' in train
    assert '"--frontres_segment_cache_k"' in train
    assert '"--frontres_segment_cache_frame_stride"' in train
    assert '"--frontres_segment_cache_max_motions"' in train
    assert '"--frontres_segment_cache_max_segments"' in train
    assert '"--frontres_segment_cache_variants_per_strength"' in train
    assert '"--frontres_segment_cache_chunk_size"' in train
    assert '"--frontres_segment_cache_perturbation_mode"' in train
    assert '"--frontres_segment_cache_perturbation_strengths"' in train
    assert '"--frontres_segment_cache_curriculum_bank_size"' in train
    assert '"--frontres_segment_cache_curriculum_frontier_scale"' in train
    assert '"--frontres_segment_cache_curriculum_dr_min"' in train
    assert '"--frontres_segment_cache_curriculum_dr_max"' in train
    assert '"--frontres_segment_cache_curriculum_progress"' in train
    assert '"--frontres_segment_cache_curriculum_seq_idx"' in train
    assert '"--frontres_segment_cache_curriculum_active_dims"' in train
    assert '"--frontres_segment_cache_curriculum_include_hard_as_train"' in train
    assert '"--frontres_segment_cache_curriculum_temporal_mode"' in train
    assert '"--frontres_segment_cache_curriculum_burst_min_steps"' in train
    assert '"--frontres_segment_cache_curriculum_burst_max_steps"' in train
    assert '"--frontres_segment_live_sentinel_only"' in train
    assert '"--frontres_segment_live_probe_only"' in train
    assert '"--frontres_segment_live_storage_write_only"' in train
    assert '"--frontres_segment_live_single_update_only"' in train
    assert '"--frontres_segment_live_update_loop_only"' in train
    assert '"--frontres_segment_live_update_steps"' in train
    assert '"--frontres_segment_shard_cache_size"' in train
    assert "Stage 3 live sentinel/probe/storage/update flags require --frontres_stage stage3_segment_hrl" in train
    assert 'if stage == "stage1_segment_cache":' in train
    assert 'elif stage in ("stage1_hsl", "stage2_hsl_warmup"):' in train
    assert 'elif stage == "stage2_acceptance":' in train
    assert 'elif stage == "stage3_segment_hrl":' in train
    assert 'FEMR_LOG_ROOT' in train
    assert 'os.path.dirname(__file__)' in train
    assert 'def _prefer_local_femr_sources()' in train
    assert '_prefer_local_femr_sources()' in train
    assert '"source/rsl_rl"' in train
    assert '"source/whole_body_tracking"' in train
    assert 'candidate_base_paths' not in train
    assert '"/workspace/"' not in train
    assert '"/hdd1/cyx/MOSAIC/"' not in train
    assert train.index("_prefer_local_femr_sources()") < train.index("from isaaclab.app import AppLauncher")
    assert train.index("_prefer_local_femr_sources()") < train.index("from whole_body_tracking.utils.my_on_policy_runner")
    assert 'stage2_authority' not in _between(
        train,
        '''parser.add_argument(
    "--frontres_stage"''',
        '''parser.add_argument(
    "--supervised_warmup_iterations"''',
    )

    stage1_cache_block = _between(train, 'if stage == "stage1_segment_cache":', '    elif stage in ("stage1_hsl", "stage2_hsl_warmup"):')
    stage1_cache_required = [
        'agent_cfg.experiment_name = "g1_flat_frontres_stage1_segment_cache"',
        'agent_cfg.max_iterations = 0',
        '_set_if_present(alg_cfg, "frontres_segment_replay_enabled", True)',
        '_set_if_present(alg_cfg, "frontres_segment_k", max(1, int(getattr(args_cli, "frontres_segment_cache_k", 4))))',
        '_set_if_present(alg_cfg, "frontres_segment_live_runner_enabled", False)',
    ]
    for needle in stage1_cache_required:
        assert needle in stage1_cache_block, needle
    assert "def _configure_frontres_stage1_segment_cache_env_cfg" in train
    assert "def _run_frontres_stage1_segment_cache(env, args_cli, log_dir: str)" in train
    assert "def _parse_frontres_segment_cache_limit(value, *, name: str) -> int | None:" in train
    assert 'raw in {"", "all", "auto", "full", "none"}' in train
    assert "return None" in _between(
        train,
        "def _parse_frontres_segment_cache_limit(value, *, name: str) -> int | None:",
        "def _frontres_segment_cache_limit_label(value: int | None) -> str:",
    )
    assert "def _frontres_stage1_motion_loader_probe(adapter, *, requested_max_motions: int | None) -> None:" in train
    assert "requested_max_motions={_frontres_segment_cache_limit_label(requested_max_motions)}" in train
    assert "[FrontRES Stage1 Segment Cache] stage1_cfg_probe" in train
    assert "[FrontRES Stage1 Segment Cache] motion_loader_probe" in train
    assert "requested multiple motions but the live motion loader loaded too few" in train
    assert "def _exit_frontres_stage1_segment_cache(env) -> None:" in train
    assert "[FrontRES Stage1 Segment Cache] auto_exit" in train
    assert "os._exit(0)" in train
    assert 'return "/hdd1/cyx/AMASS_G1Segment"' in train
    assert "[FrontRES Stage1 Segment Cache] live_sentinel" in train
    assert "FrontRESStage1EnvAdapter" in train
    assert "_frontres_stage1_motion_loader_probe(adapter, requested_max_motions=max_motions)" in train
    assert "build_stage1_segment_cache" in train
    assert "def _parse_frontres_segment_cache_active_dims(value: str) -> tuple[int, ...] | None:" in train
    assert 'choices=("hrl_curriculum_bank", "discrete_bank")' in train
    assert 'perturbation_mode={perturbation_mode}' in train
    assert 'legacy_perturbation_strengths={strengths}' in train
    assert 'curriculum_bank_size={curriculum_bank_size}' in train
    assert 'curriculum_frontier_scale={curriculum_frontier_scale}' in train
    assert 'curriculum_active_dims={curriculum_active_dims}' in train
    assert 'cache_chunk_size={cache_chunk_size}' in train
    assert "perturbation_curriculum_mode=perturbation_mode" in train
    assert "curriculum_bank_size=curriculum_bank_size" in train
    assert "curriculum_frontier_scale=curriculum_frontier_scale" in train
    assert "curriculum_dr_min=curriculum_dr_min" in train
    assert "curriculum_dr_max=curriculum_dr_max" in train
    assert "curriculum_progress=curriculum_progress" in train
    assert "curriculum_seq_idx=curriculum_seq_idx" in train
    assert "curriculum_active_dims=curriculum_active_dims" in train
    assert "curriculum_include_hard_as_train=curriculum_include_hard_as_train" in train
    assert "curriculum_temporal_mode=curriculum_temporal_mode" in train
    assert "curriculum_burst_min_steps=curriculum_burst_min_steps" in train
    assert "curriculum_burst_max_steps=curriculum_burst_max_steps" in train
    assert "cache_chunk_size=cache_chunk_size" in train
    assert "Stage 1 Segment Cache entrypoint is recognized" not in train
    assert "NotImplementedError" not in _between(
        train,
        "def _run_frontres_stage1_segment_cache(env, args_cli, log_dir: str)",
        "@hydra_task_config",
    )
    assert 'if args_cli.frontres_stage == "stage1_segment_cache":' in train
    assert train.index("_configure_frontres_stage1_segment_cache_env_cfg(env_cfg, args_cli)") < train.index("gym.make(")
    stage1_runtime_block = _between(
        train,
        'if args_cli.frontres_stage == "stage1_segment_cache":',
        "# wrap around environment for rsl-rl",
    )
    assert "_run_frontres_stage1_segment_cache(env, args_cli, log_dir)" in stage1_runtime_block
    assert "_exit_frontres_stage1_segment_cache(env)" in stage1_runtime_block
    assert stage1_runtime_block.index("_run_frontres_stage1_segment_cache(env, args_cli, log_dir)") < stage1_runtime_block.index(
        "_exit_frontres_stage1_segment_cache(env)"
    )

    stage2_block = _between(train, 'elif stage == "stage2_acceptance":', '    print(f"[FrontRES Stage]')
    required = [
        'agent_cfg.experiment_name = "g1_flat_frontres_stage2_acceptance"',
        'agent_cfg.is_full_resume = False',
        'agent_cfg.supervised_warmup_iterations = 0',
        '_set_if_present(alg_cfg, "frontres_training_objective", "hsl_hybrid")',
        '_set_if_present(alg_cfg, "frontres_acceptance_preference_weight", 1.0)',
        '_set_if_present(policy_cfg, "frontres_split_acceptance_head", True)',
        '_set_if_present(alg_cfg, "frontres_authority_actor_critic_enabled", False)',
        '_set_if_present(alg_cfg, "frontres_authority_actor_loss_weight", 0.0)',
        '_set_if_present(alg_cfg, "frontres_authority_critic_loss_weight", 0.0)',
        '_set_if_present(alg_cfg, "frontres_structured_joint_rl_enabled", False)',
        '_set_if_present(alg_cfg, "frontres_structured_joint_rl_weight", 0.0)',
        '_set_if_present(alg_cfg, "frontres_structured_joint_prior_loss_weight", 0.0)',
        '_set_if_present(policy_cfg, "frontres_authority_actor_critic", False)',
        '_set_if_present(policy_cfg, "frontres_state_router_enabled", False)',
    ]
    for needle in required:
        assert needle in stage2_block, needle
    forbidden = [
        'frontres_authority_actor_critic_enabled", True',
        'frontres_authority_actor_loss_weight", 1.0',
        'frontres_authority_critic_loss_weight", 1.0',
        'frontres_structured_joint_enabled"',
        'frontres_structured_joint_prior_weight"',
        'frontres_structured_joint_rl_enabled", True',
        'frontres_structured_joint_rl_weight", 1.0',
        'frontres_authority_return_horizon", 8',
        'frontres_perturbation_temporal_mode", "burst"',
    ]
    for needle in forbidden:
        assert needle not in stage2_block, needle

    stage3_block = _between(train, 'elif stage == "stage3_segment_hrl":', '    print(f"[FrontRES Stage]')
    stage3_required = [
        'agent_cfg.experiment_name = "g1_flat_frontres_stage3_segment_hrl"',
        'agent_cfg.is_full_resume = False',
        'agent_cfg.supervised_warmup_iterations = 0',
        '_set_if_present(alg_cfg, "frontres_training_objective", "segment_replay_hrl")',
        '_set_if_present(alg_cfg, "frontres_segment_replay_enabled", True)',
        'live_sentinel_only = live_sentinel_arg',
        'live_probe_only = live_probe_arg',
        'live_storage_only = live_storage_arg',
        'live_single_update_only = live_single_update_arg',
        'live_update_loop_only = live_update_loop_arg',
        'live_update_steps = max(1, int(getattr(args_cli, "frontres_segment_live_update_steps", 4)))',
        'live_train_enabled = not (',
        'Use only one of --frontres_segment_live_sentinel_only',
        'agent_cfg.max_iterations = 0',
        '"frontres_segment_live_runner_enabled",',
        'or live_train_enabled',
        '_set_if_present(alg_cfg, "frontres_segment_live_sentinel_only", live_sentinel_only)',
        '_set_if_present(alg_cfg, "frontres_segment_live_probe_only", live_probe_only)',
        '_set_if_present(alg_cfg, "frontres_segment_live_storage_write_only", live_storage_only)',
        '_set_if_present(alg_cfg, "frontres_segment_live_single_update_only", live_single_update_only)',
        '_set_if_present(alg_cfg, "frontres_segment_live_update_loop_only", live_update_loop_only)',
        '_set_if_present(alg_cfg, "frontres_segment_live_train_enabled", live_train_enabled)',
        '_set_if_present(alg_cfg, "frontres_segment_live_update_steps", live_update_steps)',
        '_set_if_present(alg_cfg, "frontres_hsl_init_enabled", True)',
        '_set_if_present(alg_cfg, "frontres_segment_k", 4)',
        'segment_cache_dir = getattr(args_cli, "frontres_segment_cache_dir", None) or "/hdd1/cyx/AMASS_G1Segment"',
        'shard_cache_size = max(1, int(getattr(args_cli, "frontres_segment_shard_cache_size", 8)))',
        '_set_if_present(alg_cfg, "frontres_segment_cache_dir", str(segment_cache_dir))',
        '_set_if_present(alg_cfg, "frontres_segment_shard_cache_size", shard_cache_size)',
        '_set_if_present(alg_cfg, "frontres_segment_include_boundary_diagnostic", False)',
        '_set_if_present(alg_cfg, "frontres_segment_sampler_global_frac", 0.4)',
        '_set_if_present(alg_cfg, "frontres_segment_sampler_replay_frac", 0.5)',
        '_set_if_present(alg_cfg, "frontres_segment_sampler_review_frac", 0.1)',
        '_set_if_present(alg_cfg, "frontres_segment_reset_mode", "auto")',
        '_set_if_present(alg_cfg, "frontres_acceptance_preference_weight", 0.0)',
        '_set_if_present(policy_cfg, "frontres_split_acceptance_head", False)',
        '_set_if_present(alg_cfg, "frontres_authority_actor_critic_enabled", False)',
        '_set_if_present(alg_cfg, "frontres_structured_joint_rl_enabled", False)',
    ]
    for needle in stage3_required:
        assert needle in stage3_block, needle
    stage3_forbidden = [
        'frontres_acceptance_preference_weight", 1.0',
        'frontres_split_acceptance_head", True',
        'frontres_authority_actor_critic_enabled", True',
        'frontres_structured_joint_rl_enabled", True',
    ]
    for needle in stage3_forbidden:
        assert needle not in stage3_block, needle

    algorithm_cfg = _read("source/rsl_rl/rsl_rl/modules/rsl_rl_cfg.py")
    task_cfg = _read("source/whole_body_tracking/whole_body_tracking/utils/rsl_rl_cfg.py")
    algorithm_impl = _read("source/rsl_rl/rsl_rl/algorithms/frontres_unified.py")
    runner_impl = _read("source/rsl_rl/rsl_rl/runners/on_policy_runner.py")
    live_probe_helper = _read("source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py")
    live_training_helper = _read("source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py")
    for cfg_text in (algorithm_cfg, task_cfg):
        assert "frontres_segment_replay_enabled: bool = False" in cfg_text
        assert "frontres_segment_live_runner_enabled: bool = False" in cfg_text
        assert "frontres_segment_live_sentinel_only: bool = False" in cfg_text
        assert "frontres_segment_live_probe_only: bool = False" in cfg_text
        assert "frontres_segment_live_storage_write_only: bool = False" in cfg_text
        assert "frontres_segment_live_single_update_only: bool = False" in cfg_text
        assert "frontres_segment_live_update_loop_only: bool = False" in cfg_text
        assert "frontres_segment_live_train_enabled: bool = False" in cfg_text
        assert "frontres_segment_live_update_steps: int = 4" in cfg_text
        assert "frontres_segment_live_fail_on_invalid_update: bool = True" in cfg_text
        assert "frontres_segment_live_min_valid_count: int = 1" in cfg_text
        assert "frontres_segment_live_fail_on_nonfinite: bool = True" in cfg_text
        assert "frontres_hsl_init_enabled: bool = False" in cfg_text
        assert "frontres_segment_k: int = 4" in cfg_text
        assert 'frontres_segment_cache_dir: str = ""' in cfg_text
        assert "frontres_segment_shard_cache_size: int = 8" in cfg_text
        assert "frontres_segment_include_boundary_diagnostic: bool = False" in cfg_text
        assert "frontres_segment_sampler_global_frac: float = 0.4" in cfg_text
        assert "frontres_segment_sampler_replay_frac: float = 0.5" in cfg_text
        assert "frontres_segment_sampler_review_frac: float = 0.1" in cfg_text
        assert 'frontres_segment_reset_mode: str = "auto"' in cfg_text
    assert 'frontres_training_objective == "segment_replay_hrl"' in algorithm_impl
    assert "live runner integration is disabled" in algorithm_impl
    assert "runner/PPO integration is not wired yet" in algorithm_impl
    assert "frontres_segment_live_sentinel_only" in algorithm_impl
    assert "frontres_segment_live_probe_only" in algorithm_impl
    assert "frontres_segment_live_storage_write_only" in algorithm_impl
    assert "frontres_segment_live_single_update_only" in algorithm_impl
    assert "frontres_segment_live_update_loop_only" in algorithm_impl
    assert "frontres_segment_live_train_enabled" in algorithm_impl
    assert "frontres_segment_live_update_steps" in algorithm_impl
    assert "frontres_segment_cache_dir: str = \"\"" in algorithm_impl
    assert "self.frontres_segment_cache_dir = str(frontres_segment_cache_dir or \"\")" in algorithm_impl
    assert "frontres_segment_shard_cache_size" in algorithm_impl
    assert "frontres_segment_include_boundary_diagnostic: bool = False" in algorithm_impl
    assert "self.frontres_segment_include_boundary_diagnostic = bool(frontres_segment_include_boundary_diagnostic)" in algorithm_impl
    assert "frontres_segment_live_fail_on_invalid_update" in algorithm_impl
    assert "frontres_segment_live_min_valid_count" in algorithm_impl
    assert "frontres_segment_live_fail_on_nonfinite" in algorithm_impl
    assert "runner will execute exactly one PPO optimizer step and exit" in algorithm_impl
    assert "PPO optimizer steps and exit" in algorithm_impl
    assert "PPO optimizer steps per iteration" in algorithm_impl
    assert "args_cli.frontres_segment_live_single_update_only" in train
    assert "args_cli.frontres_segment_live_update_loop_only" in train
    assert "runner.run_frontres_segment_live_update_loop(init_at_random_ep_len=True)" in train
    assert "runner.learn_frontres_segment_live(" in train
    assert "frontres_segment_live_train_enabled" in train
    assert "runner.run_frontres_segment_live_probe(init_at_random_ep_len=True)" in train
    assert "run_frontres_segment_live_probe_helper(self" in runner_impl
    assert "FrontRESSegmentRolloutStorage" not in runner_impl
    assert "FrontRESSegmentTransition" not in runner_impl
    assert "compute_frontres_segment_ppo_loss" not in runner_impl
    assert "FrontRESSegmentPPOConfig" not in runner_impl
    assert "def run_frontres_segment_live_probe" in live_probe_helper
    assert "def build_live_segment_storage" in live_probe_helper
    assert "def run_frontres_segment_single_update" in live_probe_helper
    assert "def _run_live_rollout_capture" in live_probe_helper
    assert "FrontRESSegmentRolloutStorage" in live_probe_helper
    assert "FrontRESSegmentTransition" in live_probe_helper
    assert "compute_frontres_segment_ppo_loss" in live_probe_helper
    assert "FrontRESSegmentPPOConfig" in live_probe_helper
    assert "run_frontres_segment_live_training_loop" in runner_impl
    assert "FrontRES Segment live update produced update_count=0" in live_training_helper
    assert "FrontRES Segment live update produced non-finite" in live_training_helper
    assert "too few valid PPO samples" in live_training_helper
    assert "def run_frontres_segment_live_training_loop" in live_training_helper
    assert "FrontRES Segment live update summary missing keys" in live_training_helper
    assert train.index("runner.run_frontres_segment_live_probe(init_at_random_ep_len=True)") < train.index(
        "runner.learn(num_learning_iterations=agent_cfg.max_iterations"
    )

    assert '--frontres_stage stage1_segment_cache' in stage1_cache
    assert 'MAX_MOTIONS="${MAX_MOTIONS:-all}"' in stage1_cache
    assert 'MAX_SEGMENTS="${MAX_SEGMENTS:-all}"' in stage1_cache
    assert 'CACHE_CHUNK_SIZE="${CACHE_CHUNK_SIZE:-128}"' in stage1_cache
    assert 'MAX_MOTIONS/MAX_SEGMENTS accept positive integers or all/auto.' in stage1_cache
    assert 'CACHE_CHUNK_SIZE controls how many cache records are written per payload shard.' in stage1_cache
    assert 'FRONTRES_STAGE1_PREFLIGHT_ONLY' in stage1_cache
    assert '[FrontRES Stage1 startup preflight] PASS' in stage1_cache
    assert 'Stage 1 startup preflight failed; missing cmd fragment' in stage1_cache
    assert "999999" not in stage1_cache
    assert '--frontres_segment_cache_k "${SEGMENT_K}"' in stage1_cache
    assert '--frontres_segment_cache_frame_stride "${FRAME_STRIDE}"' in stage1_cache
    assert '--frontres_segment_cache_max_motions "${MAX_MOTIONS}"' in stage1_cache
    assert '--frontres_segment_cache_max_segments "${MAX_SEGMENTS}"' in stage1_cache
    assert '--frontres_segment_cache_variants_per_strength "${VARIANTS_PER_STRENGTH}"' in stage1_cache
    assert '--frontres_segment_cache_chunk_size "${CACHE_CHUNK_SIZE}"' in stage1_cache
    assert '--frontres_segment_cache_perturbation_mode "${PERTURBATION_MODE}"' in stage1_cache
    assert '--frontres_segment_cache_perturbation_strengths "${PERTURBATION_STRENGTHS}"' in stage1_cache
    assert 'PERTURBATION_MODE="${PERTURBATION_MODE:-hrl_curriculum_bank}"' in stage1_cache
    assert 'CURRICULUM_BANK_SIZE="${CURRICULUM_BANK_SIZE:-16}"' in stage1_cache
    assert 'CURRICULUM_FRONTIER_SCALE="${CURRICULUM_FRONTIER_SCALE:-2.0}"' in stage1_cache
    assert 'CURRICULUM_DR_MIN="${CURRICULUM_DR_MIN:-1.25}"' in stage1_cache
    assert 'CURRICULUM_DR_MAX="${CURRICULUM_DR_MAX:-4.5}"' in stage1_cache
    assert 'CURRICULUM_PROGRESS="${CURRICULUM_PROGRESS:-0.8}"' in stage1_cache
    assert 'CURRICULUM_SEQ_IDX="${CURRICULUM_SEQ_IDX:-17}"' in stage1_cache
    assert 'CURRICULUM_ACTIVE_DIMS="${CURRICULUM_ACTIVE_DIMS:-0,1,2,3,4,5}"' in stage1_cache
    assert 'CURRICULUM_TEMPORAL_MODE="${CURRICULUM_TEMPORAL_MODE:-single}"' in stage1_cache
    assert 'CURRICULUM_BURST_MIN_STEPS="${CURRICULUM_BURST_MIN_STEPS:-4}"' in stage1_cache
    assert 'CURRICULUM_BURST_MAX_STEPS="${CURRICULUM_BURST_MAX_STEPS:-8}"' in stage1_cache
    assert 'VALIDATE_AFTER_BUILD="${VALIDATE_AFTER_BUILD:-1}"' in stage1_cache
    assert 'VALIDATION_EXPECT_MODE="${VALIDATION_EXPECT_MODE:-${PERTURBATION_MODE}}"' in stage1_cache
    assert 'VALIDATION_REQUIRE_BOUNDARY_DIAGNOSTIC="${VALIDATION_REQUIRE_BOUNDARY_DIAGNOSTIC:-auto}"' in stage1_cache
    assert '--frontres_segment_cache_curriculum_bank_size "${CURRICULUM_BANK_SIZE}"' in stage1_cache
    assert '--frontres_segment_cache_curriculum_frontier_scale "${CURRICULUM_FRONTIER_SCALE}"' in stage1_cache
    assert '--frontres_segment_cache_curriculum_dr_min "${CURRICULUM_DR_MIN}"' in stage1_cache
    assert '--frontres_segment_cache_curriculum_dr_max "${CURRICULUM_DR_MAX}"' in stage1_cache
    assert '--frontres_segment_cache_curriculum_progress "${CURRICULUM_PROGRESS}"' in stage1_cache
    assert '--frontres_segment_cache_curriculum_seq_idx "${CURRICULUM_SEQ_IDX}"' in stage1_cache
    assert '--frontres_segment_cache_curriculum_active_dims "${CURRICULUM_ACTIVE_DIMS}"' in stage1_cache
    assert '--frontres_segment_cache_curriculum_temporal_mode "${CURRICULUM_TEMPORAL_MODE}"' in stage1_cache
    assert '--frontres_segment_cache_curriculum_burst_min_steps "${CURRICULUM_BURST_MIN_STEPS}"' in stage1_cache
    assert '--frontres_segment_cache_curriculum_burst_max_steps "${CURRICULUM_BURST_MAX_STEPS}"' in stage1_cache
    assert 'CMD+=(--frontres_segment_cache_curriculum_include_hard_as_train)' in stage1_cache
    assert 'CACHE_DIR="${4:-/hdd1/cyx/AMASS_G1Segment}"' in stage1_cache
    assert 'g1_flat_frontres_stage1_segment_cache' in stage1_cache
    assert 'Stage 1 builds the Segment Replay cache' in stage1_cache
    assert 'After a successful build, the script validates the written cache by default.' in stage1_cache
    assert 'if [[ "${VALIDATE_AFTER_BUILD}" == "1" ]]; then' in stage1_cache
    assert 'VALIDATE_CMD=(' in stage1_cache
    assert 'source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_validator.py' in stage1_cache
    assert '--expect-mode "${VALIDATION_EXPECT_MODE}"' in stage1_cache
    assert '--min-segments "${VALIDATION_MIN_SEGMENTS}"' in stage1_cache
    assert '--min-noisy "${VALIDATION_MIN_NOISY}"' in stage1_cache
    assert '[[ "${VALIDATION_REQUIRE_BOUNDARY_DIAGNOSTIC}" == "auto" ]]' in stage1_cache
    assert '[[ "${PERTURBATION_MODE}" == "hrl_curriculum_bank" ]]' in stage1_cache
    assert '[[ "${CURRICULUM_INCLUDE_HARD_AS_TRAIN:-0}" != "1" ]]' in stage1_cache
    assert 'VALIDATE_CMD+=(--require-boundary-diagnostic)' in stage1_cache
    assert 'CACHE_DIR="${1:-/hdd1/cyx/AMASS_G1Segment}"' in stage1_cache_validator
    assert 'EXPECT_MODE="${EXPECT_MODE:-hrl_curriculum_bank}"' in stage1_cache_validator
    assert 'MIN_SEGMENTS="${MIN_SEGMENTS:-1}"' in stage1_cache_validator
    assert 'MIN_NOISY="${MIN_NOISY:-1}"' in stage1_cache_validator
    assert 'source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_validator.py' in stage1_cache_validator
    assert '--expect-mode "${EXPECT_MODE}"' in stage1_cache_validator
    assert '--min-segments "${MIN_SEGMENTS}"' in stage1_cache_validator
    assert '--min-noisy "${MIN_NOISY}"' in stage1_cache_validator
    assert 'CMD+=(--require-boundary-diagnostic)' in stage1_cache_validator
    assert '--frontres_stage stage1_hsl' in stage1
    assert 'authority' not in stage1.lower()
    assert '--frontres_stage stage2_acceptance' in stage2
    assert '--resume_student_checkpoint "${STAGE1_CHECKPOINT}"' in stage2
    assert '--is_full_resume False' in stage2
    assert 'g1_flat_frontres_stage2_acceptance' in stage2
    assert 'stage2_authority' not in stage2
    assert 'STAGE1_BUILD_ROLLOUT_CACHE="${STAGE1_BUILD_ROLLOUT_CACHE:-0}"' in root_stage1
    assert 'if [[ "${STAGE1_BUILD_ROLLOUT_CACHE}" != "1" ]]; then' in root_stage1
    assert 'STAGE1_MODE="index"' in root_stage1
    assert '"${PYTHON_BIN}" scripts/rsl_rl/build_frontres_stage1_segment_index.py' in root_stage1
    assert '--segment-k "${SEGMENT_K}"' in root_stage1
    assert '--frame-stride "${FRAME_STRIDE}"' in root_stage1
    assert '--max-motions "${MAX_MOTIONS}"' in root_stage1
    assert '--max-segments "${MAX_SEGMENTS}"' in root_stage1
    assert 'bash run/run_frontres_stage1_segment_cache.sh' in root_stage1
    assert 'NUM_ENVS="${2:-1}"' in root_stage1
    assert 'MAX_MOTIONS="${MAX_MOTIONS:-all}"' in root_stage1
    assert 'MAX_SEGMENTS="${MAX_SEGMENTS:-all}"' in root_stage1
    assert 'CACHE_CHUNK_SIZE="${CACHE_CHUNK_SIZE:-128}"' in root_stage1
    assert 'VARIANTS_PER_STRENGTH="${VARIANTS_PER_STRENGTH:-1}"' in root_stage1
    assert 'VALIDATION_MIN_SEGMENTS="${VALIDATION_MIN_SEGMENTS:-1}"' in root_stage1
    assert 'VALIDATION_MIN_NOISY="${VALIDATION_MIN_NOISY:-1}"' in root_stage1
    assert 'train_stage1_segment_cache_${STAGE1_MODE}.txt' in root_stage1
    assert 'build_rollout_cache=${STAGE1_BUILD_ROLLOUT_CACHE}' in root_stage1
    assert 'cache_chunk_size=${CACHE_CHUNK_SIZE}' in root_stage1
    assert '--frontres_stage stage2_hsl_warmup' in root_stage2
    assert 'g1_flat_frontres_stage2_hsl' in root_stage2
    assert 'SUPERVISED_WARMUP_ITERS="${SUPERVISED_WARMUP_ITERS:-${MAX_ITERS}}"' in root_stage2
    assert 'stage2_acceptance' not in root_stage2
    assert 'acceptance' not in root_stage2.lower()
    assert 'STAGE2_CHECKPOINT="$1"' in root_stage3
    assert 'bash run/run_frontres_stage3_segment_hrl.sh' in root_stage3
    assert 'CACHE_DIR="${CACHE_DIR:-/hdd1/cyx/AMASS_G1Segment}"' in root_stage3
    assert 'SHARD_CACHE_SIZE="${SHARD_CACHE_SIZE:-8}"' in root_stage3
    assert 'export CACHE_DIR' in root_stage3
    assert 'export SHARD_CACHE_SIZE' in root_stage3
    assert 'FRONTRES_STAGE_PREFLIGHT_ONLY=1' in root_stage3
    assert '[FrontRES Stage3] preflight only' in root_stage3
    assert 'train_stage3_segment_hrl.txt' in root_stage3
    assert 'stage2_acceptance' not in root_stage3
    assert 'acceptance' not in root_stage3.lower()
    assert '--frontres_stage stage3_segment_hrl' in stage3
    assert '--resume_student_checkpoint "${HSL_CHECKPOINT}"' in stage3
    assert '--is_full_resume False' in stage3
    assert 'CACHE_DIR="${CACHE_DIR:-/hdd1/cyx/AMASS_G1Segment}"' in stage3
    assert 'SHARD_CACHE_SIZE="${SHARD_CACHE_SIZE:-8}"' in stage3
    assert 'SHARD_CACHE_SIZE controls the lazy Stage 1 cache LRU size.' in stage3
    assert '--frontres_segment_cache_dir "${CACHE_DIR}"' in stage3
    assert '--frontres_segment_shard_cache_size "${SHARD_CACHE_SIZE}"' in stage3
    assert '--frontres_segment_live_update_steps "${UPDATE_STEPS}"' in stage3
    assert '" --frontres_segment_cache_dir ${CACHE_DIR} "' in stage3
    assert '" --frontres_segment_shard_cache_size ${SHARD_CACHE_SIZE} "' in stage3
    assert '--frontres_segment_live_update_loop_only' in stage3
    assert 'FRONTRES_STAGE3_RUN_CONTRACTS' in stage3
    assert 'frontres_segment_all_contract_suite.py' in stage3
    assert '[FrontRES Stage3 contract preflight] PASS' in stage3
    assert 'g1_flat_frontres_stage3_segment_hrl' in stage3
    assert 'stage2_acceptance' not in stage3
    assert 'Stage 3 loads an HSL Delta SE proposal checkpoint' in stage3
    assert '"/hdd1/cyx/MOSAIC/"' not in stage3
    assert not (ROOT / 'run/run_frontres_stage2_authority.sh').exists()
    legacy = ROOT / 'run/legacy/run_frontres_stage2_authority.sh'
    assert legacy.exists()
    assert 'Legacy ablation entrypoint' in legacy.read_text()
    print("PASS: FrontRES Stage 1/2 live presets and Stage 3 Segment Replay contract are explicit.")


if __name__ == "__main__":
    main()
