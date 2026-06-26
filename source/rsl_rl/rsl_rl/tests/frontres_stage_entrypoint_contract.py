#!/usr/bin/env python3
"""Step 9 sentinel: active FEMR entrypoints must launch HSL + acceptance, not authority."""
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
    stage1 = _read("run/run_frontres_stage1_hsl.sh")
    stage2 = _read("run/run_frontres_stage2_acceptance.sh")

    assert 'choices=("stage1_hsl", "stage2_acceptance")' in train
    assert 'elif stage == "stage2_acceptance":' in train
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
        '_set_if_present(policy_cfg, "frontres_authority_actor_critic", False)',
        '_set_if_present(policy_cfg, "frontres_state_router_enabled", False)',
    ]
    for needle in required:
        assert needle in stage2_block, needle
    forbidden = [
        'frontres_authority_actor_critic_enabled", True',
        'frontres_authority_actor_loss_weight", 1.0',
        'frontres_authority_critic_loss_weight", 1.0',
        'frontres_authority_return_horizon", 8',
        'frontres_perturbation_temporal_mode", "burst"',
    ]
    for needle in forbidden:
        assert needle not in stage2_block, needle

    assert '--frontres_stage stage1_hsl' in stage1
    assert 'authority' not in stage1.lower()
    assert '--frontres_stage stage2_acceptance' in stage2
    assert '--resume_student_checkpoint "${STAGE1_CHECKPOINT}"' in stage2
    assert '--is_full_resume False' in stage2
    assert 'g1_flat_frontres_stage2_acceptance' in stage2
    assert 'stage2_authority' not in stage2
    assert not (ROOT / 'run/run_frontres_stage2_authority.sh').exists()
    legacy = ROOT / 'run/legacy/run_frontres_stage2_authority.sh'
    assert legacy.exists()
    assert 'Legacy ablation entrypoint' in legacy.read_text()
    print("PASS: FrontRES Stage 1/2 entrypoints launch HSL proposal + acceptance training.")


if __name__ == "__main__":
    main()
