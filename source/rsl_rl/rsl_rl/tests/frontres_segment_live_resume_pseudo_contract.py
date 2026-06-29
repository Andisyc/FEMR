#!/usr/bin/env python3
"""Step 21 pseudo-parameter contract for Stage 3 live train resume.

This does not replace a server smoke run.  It proves the local runner-training
boundary can save a checkpoint, resume from that checkpoint, and keep using the
Segment Replay live training path.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


live_training_module = _load(
    "frontres_segment_live_training",
    ROOT / "rsl_rl" / "runners" / "frontres_segment_live_training.py",
)

run_frontres_segment_live_training_loop = live_training_module.run_frontres_segment_live_training_loop


def _summary(**overrides) -> dict:
    data = {
        "update_steps": 1,
        "update_count": 1,
        "ppo_valid_count": 1,
        "reward_mean": 0.25,
        "storage_valid_frac": 1.0,
        "ppo_total_loss_mean": 0.5,
        "ppo_actor_loss_mean": 0.1,
        "ppo_value_loss_mean": 0.2,
        "ppo_approx_kl_mean": 0.01,
        "ppo_clip_frac_mean": 0.0,
    }
    data.update(overrides)
    return data


class FakeBoundary:
    live_train_enabled = True


class FakeAlg:
    frontres_segment_live_fail_on_invalid_update = True
    frontres_segment_live_min_valid_count = 1
    frontres_segment_live_fail_on_nonfinite = True


class FakeRunner:
    def __init__(
        self,
        *,
        log_dir: str,
        current_learning_iteration: int = 0,
        loaded_checkpoint_path: str | None = None,
    ):
        self._frontres_segment_replay_boundary = FakeBoundary()
        self.alg = FakeAlg()
        self.current_learning_iteration = current_learning_iteration
        self.log_dir = log_dir
        self.disable_logs = False
        self.save_interval = 1
        self.update_calls: list[tuple[bool, bool]] = []
        self.saved_paths: list[str] = []
        self.probe_records: list[tuple[dict, str]] = []
        self.legacy_learn_calls = 0
        if loaded_checkpoint_path is not None:
            self._frontres_last_loaded_checkpoint_path = loaded_checkpoint_path

    def run_frontres_segment_live_update_loop(self, *, init_at_random_ep_len: bool, runner_learn: bool) -> dict:
        self.update_calls.append((init_at_random_ep_len, runner_learn))
        call_id = len(self.update_calls)
        out = _summary(
            ppo_valid_count=call_id,
            reward_mean=0.25 * call_id,
            ppo_total_loss_mean=0.5 * call_id,
        )
        print(
            "[probe step21] update_loop: "
            f"call_id={call_id} "
            f"init_at_random_ep_len={init_at_random_ep_len} "
            f"runner_learn={runner_learn} "
            f"ppo_valid_count={out['ppo_valid_count']} "
            f"reward_mean={out['reward_mean']}",
            flush=True,
        )
        return out

    def save(self, path: str) -> None:
        self.saved_paths.append(path)

    def load(self, path: str) -> None:
        self._frontres_last_loaded_checkpoint_path = path
        self.current_learning_iteration = 1

    def learn(self, *args, **kwargs) -> None:
        self.legacy_learn_calls += 1

    def _record_frontres_checkpoint_probe(self, locs: dict, checkpoint_path: str) -> None:
        self.probe_records.append((locs, checkpoint_path))


def _probe_runner(name: str, runner: FakeRunner) -> None:
    print(
        "[probe step21] "
        f"{name}: "
        f"current_learning_iteration={runner.current_learning_iteration} "
        f"loaded_checkpoint_path={getattr(runner, '_frontres_last_loaded_checkpoint_path', None)} "
        f"update_calls={runner.update_calls} "
        f"saved_paths={runner.saved_paths} "
        f"probe_record_count={len(runner.probe_records)} "
        f"legacy_learn_calls={runner.legacy_learn_calls}",
        flush=True,
    )


def test_short_training_saves_and_resume_stays_on_live_train() -> None:
    log_dir = "/tmp/frontres-step21"
    cold_runner = FakeRunner(log_dir=log_dir)
    run_frontres_segment_live_training_loop(
        cold_runner,
        num_learning_iterations=1,
        init_at_random_ep_len=True,
    )
    _probe_runner("after_cold_short_train", cold_runner)
    assert cold_runner.current_learning_iteration == 1
    assert cold_runner.update_calls == [(True, True)]
    assert cold_runner.saved_paths[-1] == f"{log_dir}/model_1.pt"
    assert cold_runner.legacy_learn_calls == 0

    resume_path = cold_runner.saved_paths[-1]
    resumed_runner = FakeRunner(log_dir=log_dir)
    resumed_runner.load(resume_path)
    _probe_runner("after_fake_load_before_resume_train", resumed_runner)
    assert resumed_runner.current_learning_iteration == 1
    assert resumed_runner._frontres_last_loaded_checkpoint_path == resume_path

    run_frontres_segment_live_training_loop(
        resumed_runner,
        num_learning_iterations=1,
        init_at_random_ep_len=True,
    )
    _probe_runner("after_resume_short_train", resumed_runner)
    assert resumed_runner.current_learning_iteration == 2
    assert resumed_runner.update_calls == [(True, True)]
    assert resumed_runner.saved_paths[-1] == f"{log_dir}/model_2.pt"
    assert resumed_runner.legacy_learn_calls == 0
    assert len(resumed_runner.probe_records) == 1


def main() -> None:
    test_short_training_saves_and_resume_stays_on_live_train()
    print("frontres_segment_live_resume_pseudo_contract: ok")


if __name__ == "__main__":
    main()
