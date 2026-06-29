# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""
import os
import argparse
import sys
import faulthandler
import signal
import threading
import traceback

os.environ.setdefault("WANDB_SILENT", "true")
# Redirect wandb local run files to HDD to avoid filling /home
os.environ.setdefault("WANDB_DIR", "/hdd0/yuxuancheng/FEMR")
os.environ.setdefault("WANDB_CACHE_DIR", "/hdd0/yuxuancheng/FEMR/.wandb_cache")


def _prefer_local_femr_sources() -> None:
    """Make this FEMR checkout win over any installed or MOSAIC source tree."""

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    for rel_path in ("source/whole_body_tracking", "source/rsl_rl"):
        source_path = os.path.join(repo_root, rel_path)
        if os.path.isdir(source_path):
            if source_path in sys.path:
                sys.path.remove(source_path)
            sys.path.insert(0, source_path)


_prefer_local_femr_sources()

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))

if WORLD_SIZE > 1:
    base = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"isaaclab_kit_{os.getuid()}")
    rank_dir = os.path.join(base, f"rank{RANK}")
    os.environ.setdefault("OMNI_USER_DIR", rank_dir)
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(rank_dir, "cache"))
    os.environ.setdefault("XDG_DATA_HOME",  os.path.join(rank_dir, "data"))
    os.environ.setdefault("XDG_CONFIG_HOME",os.path.join(rank_dir, "config"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

parser.add_argument(
    "--max_iterations", 
    type=int, 
    default=None, 
    help="RL Policy training iterations."
)

parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--frontres_stage",
    type=str,
    choices=("stage1_segment_cache", "stage1_hsl", "stage2_hsl_warmup", "stage2_acceptance", "stage3_segment_hrl"),
    default=None,
    help=(
        "Apply a FrontRES staged-training preset after Hydra config loading. "
        "This avoids fragile Hydra deep overrides such as algorithm.xxx."
    ),
)
parser.add_argument(
    "--frontres_segment_cache_dir",
    type=str,
    default=None,
    help="For Stage 1 only: directory where clean/noisy segment cache artifacts will be written.",
)
parser.add_argument(
    "--frontres_segment_cache_k",
    type=int,
    default=4,
    help="For Stage 1 only: segment rollout horizon used when building clean/noisy caches.",
)
parser.add_argument(
    "--frontres_segment_cache_frame_stride",
    type=int,
    default=1,
    help="For Stage 1 only: frame stride used when indexing motion segments.",
)
parser.add_argument(
    "--frontres_segment_cache_max_motions",
    type=int,
    default=1,
    help="For Stage 1 only: maximum motions used by the live cache sentinel/builder.",
)
parser.add_argument(
    "--frontres_segment_cache_max_segments",
    type=int,
    default=1,
    help="For Stage 1 only: maximum segments written by the live cache sentinel/builder.",
)
parser.add_argument(
    "--frontres_segment_cache_variants_per_strength",
    type=int,
    default=1,
    help="For Stage 1 only: noisy variants generated for each perturbation strength.",
)
parser.add_argument(
    "--frontres_segment_cache_perturbation_strengths",
    type=str,
    default="0.0,0.25,0.5,0.75,1.0",
    help="For Stage 1 only: comma-separated perturbation curriculum strengths for noisy caches.",
)
parser.add_argument(
    "--frontres_segment_live_sentinel_only",
    action="store_true",
    default=False,
    help="For Stage 3 only: enter the minimal live Segment Replay sentinel path without enabling PPO training.",
)
parser.add_argument(
    "--frontres_segment_live_probe_only",
    action="store_true",
    default=False,
    help="For Stage 3 only: run a short live Segment Replay rollout probe without storage writes or PPO training.",
)
parser.add_argument(
    "--frontres_segment_live_storage_write_only",
    action="store_true",
    default=False,
    help="For Stage 3 only: run a live Segment Replay probe and write independent storage without PPO training.",
)
parser.add_argument(
    "--frontres_segment_live_single_update_only",
    action="store_true",
    default=False,
    help="For Stage 3 only: run live Segment Replay storage and exactly one PPO optimizer step, then exit.",
)
parser.add_argument(
    "--frontres_segment_live_update_loop_only",
    action="store_true",
    default=False,
    help="For Stage 3 only: run a short live Segment Replay PPO update loop, then exit before normal training.",
)
parser.add_argument(
    "--frontres_segment_live_update_steps",
    type=int,
    default=4,
    help="Number of live Segment Replay PPO update steps for --frontres_segment_live_update_loop_only.",
)
parser.add_argument(
    "--supervised_warmup_iterations",
    type=int,
    default=None,
    help="Override FrontRES supervised warmup iterations before PPO starts.",
)
parser.add_argument(
    "--supervised_warmup_steps_per_iter",
    type=int,
    default=None,
    help="Override simulation steps collected per FrontRES supervised warmup iteration.",
)
parser.add_argument(
    "--supervised_warmup_max_envs_per_step",
    type=int,
    default=None,
    help="Maximum env samples kept from each warmup step for supervised SGD.",
)
parser.add_argument(
    "--is_full_resume",
    type=lambda x: str(x).lower() in ("true", "1", "yes", "y"),
    default=None,
    help=(
        "Resume mode for FrontRES checkpoints. True resumes actor+critic+optimizer+iteration; "
        "False treats the checkpoint as initialization and resets critic/optimizer/iteration."
    ),
)
parser.add_argument(
    "--frontres_debug_training",
    action="store_true",
    default=False,
    help="Enable the shortened FrontRES debug schedule for reward/DR tuning.",
)
parser.add_argument(
    "--frontres_eval_dr_sweep",
    action="store_true",
    default=False,
    help="Run an evaluation-only fixed-DR sweep after loading the checkpoint, then exit without training.",
)
parser.add_argument(
    "--frontres_eval_dr_scales",
    type=str,
    default="1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0",
    help="Comma-separated fixed dr_scale values for FrontRES-vs-GMT stress evaluation.",
)
parser.add_argument(
    "--frontres_eval_iterations_per_scale",
    type=int,
    default=20,
    help="Number of rollout iterations collected per fixed dr_scale during evaluation.",
)

# single motion for testing
# motion_path = '/home/chengyuxuan/MOSAIC/motion_npz/dance1_subject1.npz'

parser.add_argument("--motion", type=str, default=None, help="motion or motion file path.") # required=True, 

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

faulthandler.enable()
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    print("[DEBUG] Registered SIGUSR1 stack dump. Use: kill -USR1 <pid>", flush=True)


def _dump_uncaught_exception(exc_type, exc_value, exc_tb):
    print("\n[Train] Uncaught exception in main thread:", flush=True)
    traceback.print_exception(exc_type, exc_value, exc_tb)
    print("\n[Train] Python stack dump for all live threads:", flush=True)
    faulthandler.dump_traceback(all_threads=True)


def _dump_thread_exception(args):
    print(f"\n[Train] Uncaught exception in thread {args.thread.name}:", flush=True)
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
    print("\n[Train] Python stack dump for all live threads:", flush=True)
    faulthandler.dump_traceback(all_threads=True)


sys.excepthook = _dump_uncaught_exception
threading.excepthook = _dump_thread_exception

if WORLD_SIZE > 1:
    args_cli.distributed = True
if args_cli.distributed:
    args_cli.device = f"cuda:{LOCAL_RANK}"
if args_cli.distributed and RANK != 0:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    args_cli.video = False

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

if (
    sys.platform.startswith("linux")
    and not args_cli.headless
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    print("[Train] No display detected; forcing --headless for Isaac Sim startup.", flush=True)
    args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
print(
    f"[Train] Launching Isaac Sim: device={args_cli.device}, "
    f"headless={args_cli.headless}, enable_cameras={args_cli.enable_cameras}",
    flush=True,
)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[Train] Isaac Sim ready.", flush=True)

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import torch
from datetime import datetime
import numpy as np

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils import configclass
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.assets.articulation.articulation import Articulation

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _patch_articulation_update_until_data_ready() -> None:
    """Guard IsaacLab startup scene.update() before ArticulationData exists."""

    if getattr(Articulation, "_mosaic_safe_update_patched", False):
        return
    _original_update = Articulation.update

    def _safe_update(self, dt):
        if not hasattr(self, "_data"):
            return
        return _original_update(self, dt)

    Articulation.update = _safe_update
    Articulation._mosaic_safe_update_patched = True


_patch_articulation_update_until_data_ready()


@configclass
class _NoOpCfg:
    pass


def _sanitize_env_cfg_for_training(env_cfg) -> None:
    """Avoid IsaacLab startup callbacks failing on None managers/debug visuals."""

    # Some FrontRES configs intentionally disable managers with None, but
    # IsaacLab manager callbacks still assume cfg has a __dict__ during reset.
    for field in ("events", "curriculum"):
        if hasattr(env_cfg, field) and getattr(env_cfg, field) is None:
            setattr(env_cfg, field, _NoOpCfg())

    # Headless/multi-GPU training does not need debug visualization.  Leaving
    # these enabled can register callbacks before Articulation data is ready.
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is not None and hasattr(motion_cfg, "debug_vis"):
        motion_cfg.debug_vis = False
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "contact_forces"):
        env_cfg.scene.contact_forces.debug_vis = False

    # Large FrontRES runs can create hundreds of thousands of rigid bodies and
    # contacts.  If the PhysX GPU capacities are left at defaults, Omniverse can
    # fail to create the tensor simulation view, which later appears as missing
    # Articulation._data.  Scale the common capacities with num_envs.
    physx = getattr(getattr(env_cfg, "sim", None), "physx", None)
    if physx is not None and hasattr(env_cfg, "scene"):
        num_envs = int(getattr(env_cfg.scene, "num_envs", 0) or 0)

        def _raise_capacity(name: str, value: int) -> None:
            if hasattr(physx, name):
                current = getattr(physx, name)
                if current is None or int(current) < int(value):
                    setattr(physx, name, int(value))

        _raise_capacity("gpu_max_rigid_contact_count", max(2**23, num_envs * 4096))
        _raise_capacity("gpu_max_rigid_patch_count", max(15 * 2**17, num_envs * 512))
        _raise_capacity("gpu_found_lost_pairs_capacity", max(2**24, num_envs * 2048))
        _raise_capacity("gpu_found_lost_aggregate_pairs_capacity", max(2**23, num_envs * 1024))
        _raise_capacity("gpu_total_aggregate_pairs_capacity", max(2**23, num_envs * 1024))
        _raise_capacity("gpu_collision_stack_size", max(2**26, num_envs * 8192))
        _raise_capacity("gpu_heap_capacity", max(2**27, num_envs * 16384))
        _raise_capacity("gpu_temp_buffer_capacity", max(2**26, num_envs * 8192))

        print(
            "[INFO] PhysX GPU capacities prepared for "
            f"{num_envs} envs: "
            f"contact={getattr(physx, 'gpu_max_rigid_contact_count', 'N/A')}, "
            f"patch={getattr(physx, 'gpu_max_rigid_patch_count', 'N/A')}, "
            f"pairs={getattr(physx, 'gpu_found_lost_pairs_capacity', 'N/A')}",
            flush=True,
        )


def _configure_frontres_motion_perturbations(env_cfg, agent_cfg) -> None:
    """Align motion perturbation channels with the FrontRES action mask."""
    if not hasattr(env_cfg, "motion_perturbations"):
        return
    mode = str(getattr(agent_cfg, "frontres_perturbation_channels", "all")).lower()
    pt = env_cfg.motion_perturbations

    if mode in ("all", "composite", "full"):
        # Explicitly mirror the agent-side full-output test settings into the
        # environment perturbation config.  Without this, "all" silently falls
        # back to whatever the task cfg default happens to contain, making it
        # hard to tell whether the joint test is really exercising every
        # controllable channel.
        for name in (
            "float_prob",
            "float_ratio",
            "sink_prob",
            "sink_ratio",
            "foot_slip_prob",
            "foot_slip_ratio",
            "lateral_drift_prob",
            "lateral_drift_std",
            "root_tilt_prob",
            "root_tilt_max_rad",
            "joint_noise_prob",
            "joint_noise_std",
            "iid_prob_z",
            "iid_std_z",
            "iid_prob_xy",
            "iid_std_xy",
            "iid_prob_rp",
            "iid_std_rp",
            "iid_prob_ya",
            "iid_std_ya",
            "local_root_artifact_prob",
            "local_root_artifact_xy_std",
            "local_root_artifact_yaw_std",
        ):
            if hasattr(agent_cfg, name) and hasattr(pt, name):
                setattr(pt, name, type(getattr(pt, name))(getattr(agent_cfg, name)))
        for name in ("local_root_artifact_min_steps", "local_root_artifact_max_steps"):
            if hasattr(agent_cfg, name) and hasattr(pt, name):
                setattr(pt, name, int(getattr(agent_cfg, name)))
        print(
            "[INFO] FrontRES perturbation alignment: all "
            f"(float={pt.float_prob}/{pt.float_ratio}, sink={pt.sink_prob}/{pt.sink_ratio}, "
            f"foot_slip={pt.foot_slip_prob}/{pt.foot_slip_ratio}, "
            f"lateral={pt.lateral_drift_prob}/{pt.lateral_drift_std}, "
            f"root_tilt={pt.root_tilt_prob}/{pt.root_tilt_max_rad}, "
            f"iid_xy={pt.iid_prob_xy}/{pt.iid_std_xy}, iid_z={pt.iid_prob_z}/{pt.iid_std_z}, "
            f"iid_rp={pt.iid_prob_rp}/{pt.iid_std_rp}, iid_yaw={pt.iid_prob_ya}/{pt.iid_std_ya}, "
            f"local_artifact={pt.local_root_artifact_prob}/"
            f"{pt.local_root_artifact_xy_std}/{pt.local_root_artifact_yaw_std}/"
            f"{pt.local_root_artifact_min_steps}-{pt.local_root_artifact_max_steps})",
            flush=True,
        )
        return
    if mode not in (
        "xy_yaw", "xy-yaw", "xyyaw",
        "z_rp", "z-rp", "zrp", "rp_z", "rp-z", "rpz", "vertical_contact",
        "rp", "local_rp", "rp_only", "strong_rp",
    ):
        raise ValueError(
            "frontres_perturbation_channels must be one of "
            "{'all', 'composite', 'full', 'xy_yaw', 'z_rp', 'rp_z', 'vertical_contact', 'rp'}; got "
            f"{mode!r}."
        )

    # Disable all generic channels first, then re-enable only the channels
    # controllable by the selected FrontRES task-space action mask.
    pt.float_prob = 0.0
    pt.float_ratio = 0.0
    pt.sink_prob = 0.0
    pt.sink_ratio = 0.0
    pt.foot_slip_prob = 0.0
    pt.foot_slip_ratio = 0.0
    pt.lateral_drift_prob = 0.0
    pt.lateral_drift_std = 0.0
    pt.root_tilt_prob = 0.0
    pt.root_tilt_max_rad = 0.0
    pt.joint_noise_prob = 0.0
    pt.joint_noise_std = 0.0
    pt.iid_prob_z = 0.0
    pt.iid_std_z = 0.0
    pt.iid_prob_rp = 0.0
    pt.iid_std_rp = 0.0

    if mode in ("xy_yaw", "xy-yaw", "xyyaw"):
        # X/Y/Yaw are injected as short local root artifacts, not global drift:
        # a brief anchor jump breaks contact/heading consistency and gives both
        # supervised warmup and PPO a clear executable signal.
        pt.iid_prob_xy = float(getattr(agent_cfg, "iid_prob_xy", pt.iid_prob_xy))
        pt.iid_std_xy = float(getattr(agent_cfg, "iid_std_xy", pt.iid_std_xy))
        pt.iid_prob_ya = float(getattr(agent_cfg, "iid_prob_ya", pt.iid_prob_ya))
        pt.iid_std_ya = float(getattr(agent_cfg, "iid_std_ya", pt.iid_std_ya))
        pt.local_root_artifact_prob = float(getattr(
            agent_cfg, "local_root_artifact_prob", getattr(pt, "local_root_artifact_prob", 0.0)))
        pt.local_root_artifact_min_steps = int(getattr(
            agent_cfg, "local_root_artifact_min_steps", getattr(pt, "local_root_artifact_min_steps", 3)))
        pt.local_root_artifact_max_steps = int(getattr(
            agent_cfg, "local_root_artifact_max_steps", getattr(pt, "local_root_artifact_max_steps", 8)))
        pt.local_root_artifact_xy_std = float(getattr(
            agent_cfg, "local_root_artifact_xy_std", getattr(pt, "local_root_artifact_xy_std", 0.0)))
        pt.local_root_artifact_yaw_std = float(getattr(
            agent_cfg, "local_root_artifact_yaw_std", getattr(pt, "local_root_artifact_yaw_std", 0.0)))
        print(
            "[INFO] FrontRES perturbation alignment: xy_yaw "
            f"(iid_xy={pt.iid_prob_xy}/{pt.iid_std_xy}, iid_yaw={pt.iid_prob_ya}/{pt.iid_std_ya}; "
            f"local_artifact={pt.local_root_artifact_prob}/"
            f"{pt.local_root_artifact_xy_std}/{pt.local_root_artifact_yaw_std}/"
            f"{pt.local_root_artifact_min_steps}-{pt.local_root_artifact_max_steps}; "
            "z/rp/joint disabled)",
            flush=True,
        )
        return

    if mode in ("rp", "local_rp", "rp_only", "strong_rp"):
        # RP-only specialist: isolate roll/pitch anchor artifacts so we can
        # measure GMT's angular robustness limit without z-contact coupling.
        pt.root_tilt_prob = float(getattr(agent_cfg, "root_tilt_prob", 0.3))
        pt.root_tilt_max_rad = float(getattr(agent_cfg, "root_tilt_max_rad", 0.05))
        pt.iid_prob_rp = float(getattr(agent_cfg, "iid_prob_rp", pt.iid_prob_rp))
        pt.iid_std_rp = float(getattr(agent_cfg, "iid_std_rp", pt.iid_std_rp))
        print(
            "[INFO] FrontRES perturbation alignment: rp "
            f"(root_tilt={pt.root_tilt_prob}/{pt.root_tilt_max_rad}, "
            f"iid_rp={pt.iid_prob_rp}/{pt.iid_std_rp}; "
            "xy/yaw/z/joint disabled)",
            flush=True,
        )
        return

    # Z/Roll/Pitch experiment: only vertical float/sink and root tilt/IID
    # perturbations are enabled, matching active dims [dz, droll, dpitch].
    pt.float_prob = float(getattr(agent_cfg, "float_prob", 0.3))
    pt.float_ratio = float(getattr(agent_cfg, "float_ratio", 0.05))
    pt.sink_prob = float(getattr(agent_cfg, "sink_prob", 0.3))
    pt.sink_ratio = float(getattr(agent_cfg, "sink_ratio", 0.04))
    pt.root_tilt_prob = float(getattr(agent_cfg, "root_tilt_prob", 0.3))
    pt.root_tilt_max_rad = float(getattr(agent_cfg, "root_tilt_max_rad", 0.05))
    pt.iid_prob_z = float(getattr(agent_cfg, "iid_prob_z", pt.iid_prob_z))
    pt.iid_std_z = float(getattr(agent_cfg, "iid_std_z", pt.iid_std_z))
    pt.iid_prob_rp = float(getattr(agent_cfg, "iid_prob_rp", pt.iid_prob_rp))
    pt.iid_std_rp = float(getattr(agent_cfg, "iid_std_rp", pt.iid_std_rp))
    print(
        "[INFO] FrontRES perturbation alignment: z_rp "
        f"(float={pt.float_prob}/{pt.float_ratio}, sink={pt.sink_prob}/{pt.sink_ratio}, "
        f"root_tilt={pt.root_tilt_prob}/{pt.root_tilt_max_rad}, "
        f"iid_z={pt.iid_prob_z}/{pt.iid_std_z}, iid_rp={pt.iid_prob_rp}/{pt.iid_std_rp}; "
        "xy/yaw/local/joint disabled)",
        flush=True,
    )


def _set_if_present(obj, name: str, value) -> None:
    if obj is not None and hasattr(obj, name):
        setattr(obj, name, value)


def _apply_frontres_stage_preset(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli) -> None:
    """Apply FrontRES staged-training presets through Python config objects, not Hydra overrides."""

    stage = getattr(args_cli, "frontres_stage", None)
    live_sentinel_arg = bool(getattr(args_cli, "frontres_segment_live_sentinel_only", False))
    live_probe_arg = bool(getattr(args_cli, "frontres_segment_live_probe_only", False))
    live_storage_arg = bool(getattr(args_cli, "frontres_segment_live_storage_write_only", False))
    live_single_update_arg = bool(getattr(args_cli, "frontres_segment_live_single_update_only", False))
    live_update_loop_arg = bool(getattr(args_cli, "frontres_segment_live_update_loop_only", False))
    if (
        live_sentinel_arg
        or live_probe_arg
        or live_storage_arg
        or live_single_update_arg
        or live_update_loop_arg
    ) and stage != "stage3_segment_hrl":
        raise ValueError("Stage 3 live sentinel/probe/storage/update flags require --frontres_stage stage3_segment_hrl.")
    if stage is None:
        return

    alg_cfg = getattr(agent_cfg, "algorithm", None)
    policy_cfg = getattr(agent_cfg, "policy", None)
    if alg_cfg is None:
        raise AttributeError("--frontres_stage requires an agent config with an algorithm section.")

    if stage == "stage1_segment_cache":
        if getattr(args_cli, "experiment_name", None) is None:
            agent_cfg.experiment_name = "g1_flat_frontres_stage1_segment_cache"
        agent_cfg.max_iterations = 0
        _set_if_present(agent_cfg, "frontres_stage1_exit_after_warmup", False)
        _set_if_present(alg_cfg, "frontres_segment_replay_enabled", True)
        _set_if_present(alg_cfg, "frontres_segment_k", max(1, int(getattr(args_cli, "frontres_segment_cache_k", 4))))
        _set_if_present(alg_cfg, "frontres_segment_live_runner_enabled", False)
    elif stage in ("stage1_hsl", "stage2_hsl_warmup"):
        if getattr(args_cli, "experiment_name", None) is None:
            agent_cfg.experiment_name = "g1_flat_frontres_stage2_hsl"
        _set_if_present(agent_cfg, "frontres_stage1_exit_after_warmup", True)
        _set_if_present(alg_cfg, "frontres_training_objective", "supervised_restore")
        _set_if_present(alg_cfg, "lambda_supervised", 1.0)
        _set_if_present(alg_cfg, "lambda_supervised_min", 1.0)
        _set_if_present(alg_cfg, "frontres_authority_actor_critic_enabled", False)
        _set_if_present(alg_cfg, "frontres_authority_actor_loss_weight", 0.0)
        _set_if_present(alg_cfg, "frontres_authority_critic_loss_weight", 0.0)
        _set_if_present(policy_cfg, "frontres_authority_actor_critic", False)
        _set_if_present(agent_cfg, "critic_warmup_iterations", 0)
        _set_if_present(agent_cfg, "ppo_actor_warmup_iterations", 1_000_000)
        _set_if_present(agent_cfg, "ppo_actor_ramp_iterations", 0)
    elif stage == "stage2_acceptance":
        if getattr(args_cli, "experiment_name", None) is None:
            agent_cfg.experiment_name = "g1_flat_frontres_stage2_acceptance"
        if getattr(args_cli, "is_full_resume", None) is None:
            agent_cfg.is_full_resume = False
        _set_if_present(agent_cfg, "frontres_stage1_exit_after_warmup", False)
        agent_cfg.supervised_warmup_iterations = 0
        _set_if_present(alg_cfg, "frontres_training_objective", "hsl_hybrid")
        _set_if_present(alg_cfg, "lambda_supervised", 0.20)
        _set_if_present(alg_cfg, "lambda_supervised_min", 0.20)
        _set_if_present(alg_cfg, "lambda_supervised_decay", 1.0)
        _set_if_present(alg_cfg, "frontres_acceptance_preference_weight", 1.0)
        _set_if_present(alg_cfg, "frontres_state_alpha_weight", 0.0)
        _set_if_present(alg_cfg, "frontres_authority_actor_critic_enabled", False)
        _set_if_present(alg_cfg, "frontres_authority_actor_loss_weight", 0.0)
        _set_if_present(alg_cfg, "frontres_authority_critic_loss_weight", 0.0)
        _set_if_present(alg_cfg, "frontres_structured_joint_rl_enabled", False)
        _set_if_present(alg_cfg, "frontres_structured_joint_rl_weight", 0.0)
        _set_if_present(alg_cfg, "frontres_structured_joint_prior_loss_weight", 0.0)
        _set_if_present(policy_cfg, "frontres_split_acceptance_head", True)
        _set_if_present(policy_cfg, "frontres_authority_actor_critic", False)
        _set_if_present(policy_cfg, "frontres_state_router_enabled", False)
        _set_if_present(agent_cfg, "critic_warmup_iterations", 0)
        _set_if_present(agent_cfg, "ppo_actor_warmup_iterations", 0)
        _set_if_present(agent_cfg, "ppo_actor_ramp_iterations", 0)
        _set_if_present(agent_cfg, "frontres_perturbation_temporal_mode", "single")
        _set_if_present(agent_cfg, "frontres_perturbation_burst_min_steps", 1)
        _set_if_present(agent_cfg, "frontres_perturbation_burst_max_steps", 1)
    elif stage == "stage3_segment_hrl":
        if getattr(args_cli, "experiment_name", None) is None:
            agent_cfg.experiment_name = "g1_flat_frontres_stage3_segment_hrl"
        if getattr(args_cli, "is_full_resume", None) is None:
            agent_cfg.is_full_resume = False
        _set_if_present(agent_cfg, "frontres_stage1_exit_after_warmup", False)
        agent_cfg.supervised_warmup_iterations = 0
        live_sentinel_only = live_sentinel_arg
        live_probe_only = live_probe_arg
        live_storage_only = live_storage_arg
        live_single_update_only = live_single_update_arg
        live_update_loop_only = live_update_loop_arg
        live_update_steps = max(1, int(getattr(args_cli, "frontres_segment_live_update_steps", 4)))
        live_train_enabled = not (
            live_sentinel_only
            or live_probe_only
            or live_storage_only
            or live_single_update_only
            or live_update_loop_only
        )
        if sum((live_sentinel_only, live_probe_only, live_storage_only, live_single_update_only, live_update_loop_only)) > 1:
            raise ValueError(
                "Use only one of --frontres_segment_live_sentinel_only, "
                "--frontres_segment_live_probe_only, --frontres_segment_live_storage_write_only, "
                "--frontres_segment_live_single_update_only, or --frontres_segment_live_update_loop_only."
            )
        if live_sentinel_only or live_probe_only or live_storage_only or live_single_update_only or live_update_loop_only:
            agent_cfg.max_iterations = 0
        _set_if_present(alg_cfg, "frontres_training_objective", "segment_replay_hrl")
        _set_if_present(alg_cfg, "frontres_segment_replay_enabled", True)
        _set_if_present(
            alg_cfg,
            "frontres_segment_live_runner_enabled",
            (
                live_sentinel_only
                or live_probe_only
                or live_storage_only
                or live_single_update_only
                or live_update_loop_only
                or live_train_enabled
            ),
        )
        _set_if_present(alg_cfg, "frontres_segment_live_sentinel_only", live_sentinel_only)
        _set_if_present(alg_cfg, "frontres_segment_live_probe_only", live_probe_only)
        _set_if_present(alg_cfg, "frontres_segment_live_storage_write_only", live_storage_only)
        _set_if_present(alg_cfg, "frontres_segment_live_single_update_only", live_single_update_only)
        _set_if_present(alg_cfg, "frontres_segment_live_update_loop_only", live_update_loop_only)
        _set_if_present(alg_cfg, "frontres_segment_live_train_enabled", live_train_enabled)
        _set_if_present(alg_cfg, "frontres_segment_live_update_steps", live_update_steps)
        _set_if_present(alg_cfg, "frontres_hsl_init_enabled", True)
        _set_if_present(alg_cfg, "frontres_segment_k", 4)
        _set_if_present(alg_cfg, "frontres_segment_sampler_global_frac", 0.4)
        _set_if_present(alg_cfg, "frontres_segment_sampler_replay_frac", 0.5)
        _set_if_present(alg_cfg, "frontres_segment_sampler_review_frac", 0.1)
        _set_if_present(alg_cfg, "frontres_segment_reset_mode", "auto")
        _set_if_present(alg_cfg, "frontres_acceptance_preference_weight", 0.0)
        _set_if_present(alg_cfg, "frontres_state_alpha_weight", 0.0)
        _set_if_present(alg_cfg, "frontres_authority_actor_critic_enabled", False)
        _set_if_present(alg_cfg, "frontres_authority_actor_loss_weight", 0.0)
        _set_if_present(alg_cfg, "frontres_authority_critic_loss_weight", 0.0)
        _set_if_present(alg_cfg, "frontres_structured_joint_rl_enabled", False)
        _set_if_present(alg_cfg, "frontres_structured_joint_rl_weight", 0.0)
        _set_if_present(alg_cfg, "frontres_structured_joint_prior_loss_weight", 0.0)
        _set_if_present(policy_cfg, "frontres_split_acceptance_head", False)
        _set_if_present(policy_cfg, "frontres_authority_actor_critic", False)
        _set_if_present(policy_cfg, "frontres_state_router_enabled", False)
        _set_if_present(agent_cfg, "critic_warmup_iterations", 0)
        _set_if_present(agent_cfg, "ppo_actor_warmup_iterations", 0)
        _set_if_present(agent_cfg, "ppo_actor_ramp_iterations", 0)

    print(f"[FrontRES Stage] Applied preset: {stage}", flush=True)
    print(
        "[FrontRES Stage] "
        f"experiment={getattr(agent_cfg, 'experiment_name', 'n/a')}, "
        f"objective={getattr(alg_cfg, 'frontres_training_objective', 'n/a')}, "
        f"segment_replay={getattr(alg_cfg, 'frontres_segment_replay_enabled', 'n/a')}, "
        f"segment_live={getattr(alg_cfg, 'frontres_segment_live_runner_enabled', 'n/a')}, "
        f"segment_sentinel={getattr(alg_cfg, 'frontres_segment_live_sentinel_only', 'n/a')}, "
        f"segment_probe={getattr(alg_cfg, 'frontres_segment_live_probe_only', 'n/a')}, "
        f"segment_storage={getattr(alg_cfg, 'frontres_segment_live_storage_write_only', 'n/a')}, "
        f"segment_single_update={getattr(alg_cfg, 'frontres_segment_live_single_update_only', 'n/a')}, "
        f"segment_update_loop={getattr(alg_cfg, 'frontres_segment_live_update_loop_only', 'n/a')}, "
        f"segment_train={getattr(alg_cfg, 'frontres_segment_live_train_enabled', 'n/a')}, "
        f"segment_update_steps={getattr(alg_cfg, 'frontres_segment_live_update_steps', 'n/a')}, "
        f"segment_k={getattr(alg_cfg, 'frontres_segment_k', 'n/a')}, "
        f"authority={getattr(alg_cfg, 'frontres_authority_actor_critic_enabled', 'n/a')}, "
        f"structured_joint={getattr(alg_cfg, 'frontres_structured_joint_rl_enabled', 'n/a')}/"
        f"{getattr(alg_cfg, 'frontres_structured_joint_rl_weight', 'n/a')}, "
        f"max_iterations={getattr(agent_cfg, 'max_iterations', 'n/a')}, "
        f"supervised_warmup={getattr(agent_cfg, 'supervised_warmup_iterations', 'n/a')}, "
        f"is_full_resume={getattr(agent_cfg, 'is_full_resume', 'n/a')}",
        flush=True,
    )


def _parse_frontres_segment_cache_strengths(value: str) -> list[float]:
    strengths: list[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        strengths.append(float(item))
    if not strengths:
        raise ValueError("--frontres_segment_cache_perturbation_strengths must contain at least one value.")
    if any(strength < 0.0 for strength in strengths):
        raise ValueError("--frontres_segment_cache_perturbation_strengths must be non-negative.")
    return strengths


def _frontres_stage1_segment_cache_dir(args_cli, log_dir: str) -> str:
    cache_dir = getattr(args_cli, "frontres_segment_cache_dir", None)
    if cache_dir:
        return os.path.abspath(str(cache_dir))
    return "/hdd1/cyx/AMASS_G1Segment"


def _configure_frontres_stage1_segment_cache_env_cfg(env_cfg, args_cli) -> None:
    motion_cfg = getattr(getattr(env_cfg, "commands", None), "motion", None)
    if motion_cfg is None:
        return
    max_motions = max(1, int(getattr(args_cli, "frontres_segment_cache_max_motions", 1)))
    if hasattr(motion_cfg, "motion_dataset_shard_across_gpus"):
        motion_cfg.motion_dataset_shard_across_gpus = False
    if hasattr(motion_cfg, "motion_dataset_load_cap"):
        motion_cfg.motion_dataset_load_cap = max_motions
    if hasattr(motion_cfg, "motion_dataset_log_shard_info"):
        motion_cfg.motion_dataset_log_shard_info = True
    if hasattr(motion_cfg, "resample_motions_every_s"):
        motion_cfg.resample_motions_every_s = 1.0e9
    zero_ranges = {name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")}
    if hasattr(motion_cfg, "pose_range"):
        motion_cfg.pose_range = dict(zero_ranges)
    if hasattr(motion_cfg, "velocity_range"):
        motion_cfg.velocity_range = dict(zero_ranges)
    if hasattr(motion_cfg, "joint_position_range"):
        motion_cfg.joint_position_range = (0.0, 0.0)


def _run_frontres_stage1_segment_cache(env, args_cli, log_dir: str) -> None:
    cache_dir = _frontres_stage1_segment_cache_dir(args_cli, log_dir)
    segment_k = max(1, int(getattr(args_cli, "frontres_segment_cache_k", 4)))
    frame_stride = max(1, int(getattr(args_cli, "frontres_segment_cache_frame_stride", 1)))
    max_motions = max(1, int(getattr(args_cli, "frontres_segment_cache_max_motions", 1)))
    max_segments = max(1, int(getattr(args_cli, "frontres_segment_cache_max_segments", 1)))
    variants_per_strength = max(1, int(getattr(args_cli, "frontres_segment_cache_variants_per_strength", 1)))
    strengths = _parse_frontres_segment_cache_strengths(
        getattr(args_cli, "frontres_segment_cache_perturbation_strengths", "0.0,0.25,0.5,0.75,1.0")
    )
    print(
        "[FrontRES Stage1 Segment Cache] live_sentinel "
        f"stage={getattr(args_cli, 'frontres_stage', None)} "
        f"motion={getattr(args_cli, 'motion', None)} "
        f"cache_dir={cache_dir} "
        f"segment_k={segment_k} "
        f"frame_stride={frame_stride} "
        f"max_motions={max_motions} "
        f"max_segments={max_segments} "
        f"variants_per_strength={variants_per_strength} "
        f"perturbation_strengths={strengths}",
        flush=True,
    )
    from rsl_rl.frontres.frontres_segment_cache_builder import (
        FrontRESStage1CacheBuilderConfig,
        build_stage1_segment_cache,
    )
    from rsl_rl.frontres.frontres_segment_stage1_env_hooks import FrontRESStage1EnvAdapter

    adapter = FrontRESStage1EnvAdapter(
        env,
        amass_root=str(getattr(args_cli, "motion", "")),
        trace=True,
        baseline_rollout_steps=segment_k,
    )
    result = build_stage1_segment_cache(
        adapter,
        FrontRESStage1CacheBuilderConfig(
            amass_root=str(getattr(args_cli, "motion", "")),
            cache_dir=cache_dir,
            horizon_k=segment_k,
            frame_stride=frame_stride,
            max_motions=max_motions,
            max_segments=max_segments,
            strengths=tuple(float(item) for item in strengths),
            variants_per_strength=variants_per_strength,
            base_seed=int(getattr(args_cli, "seed", 0) or 0),
            env_id=0,
        ),
    )
    print(
        "[FrontRES Stage1 Segment Cache] result "
        f"segment_count={result.segment_count} "
        f"clean_count={result.clean_count} "
        f"noisy_count={result.noisy_count} "
        f"strength_counts={result.strength_counts} "
        f"segment_index_path={result.segment_index_path} "
        f"clean_shard_path={result.clean_shard_path} "
        f"noisy_shard_paths={result.noisy_shard_paths} "
        f"metadata_path={result.metadata_path}",
        flush=True,
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point") # 
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations)
    if args_cli.supervised_warmup_iterations is not None:
        agent_cfg.supervised_warmup_iterations = args_cli.supervised_warmup_iterations
    if args_cli.supervised_warmup_steps_per_iter is not None:
        agent_cfg.supervised_warmup_steps_per_iter = args_cli.supervised_warmup_steps_per_iter
    if args_cli.supervised_warmup_max_envs_per_step is not None:
        agent_cfg.supervised_warmup_max_envs_per_step = args_cli.supervised_warmup_max_envs_per_step
    if args_cli.is_full_resume is not None:
        agent_cfg.is_full_resume = args_cli.is_full_resume
    if args_cli.frontres_debug_training:
        agent_cfg.frontres_debug_training = True
    _apply_frontres_stage_preset(agent_cfg, args_cli)

    # set seeds (explicit rank offset for distributed to avoid identical sampling across ranks)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    base_seed = int(agent_cfg.seed)
    rank = int(os.environ.get("RANK", "0"))

    # stride avoids overlaps if some components use multiple RNG draws per step
    seed_stride = int(os.environ.get("SEED_STRIDE", "1000"))
    env_seed = base_seed + rank * seed_stride
    env_cfg.seed = env_seed

    # also seed common RNGs to keep per-rank randomness independent
    random.seed(env_seed)
    np.random.seed(env_seed)
    torch.manual_seed(env_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(env_seed)

    # specify device
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        print(f"[INFO] Distributed seeding: base_seed={base_seed}, rank={rank}, env_seed={env_seed} (stride={seed_stride})")
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = args_cli.device if args_cli.device is not None else agent_cfg.device

    # load in motion sequence
    env_cfg.commands.motion.motion = args_cli.motion
    if args_cli.frontres_stage == "stage1_segment_cache":
        _configure_frontres_stage1_segment_cache_env_cfg(env_cfg, args_cli)
    _configure_frontres_motion_perturbations(env_cfg, agent_cfg)
    _sanitize_env_cfg_for_training(env_cfg)

    # specify directory for logging experiments
    # log_root_path 根据 experiment_name 自动派生，避免不同训练阶段的 checkpoint 混入同一目录。
    # 默认绑定当前 FEMR checkout；FEMR_LOG_ROOT 可显式覆盖到服务器磁盘路径。
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    base_path = os.path.abspath(os.environ.get("FEMR_LOG_ROOT", repo_root))

    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"FEMR log root does not exist: {base_path}")

    log_root_path = os.path.join(base_path, agent_cfg.experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.frontres_stage == "stage1_segment_cache":
        _run_frontres_stage1_segment_cache(env, args_cli, log_dir)
        env.close()
        return

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # If student_checkpoint_path is set as an absolute path in the config, use it directly.
        # This bypasses get_checkpoint_path() which only looks inside the current experiment's
        # log_root_path — cross-experiment loading (e.g. Stage 1 → Stage 2) requires this.
        _direct = getattr(agent_cfg, "student_checkpoint_path", None)
        if _direct is not None and os.path.isfile(str(_direct)):
            resume_path = str(_direct)
            print(f"[INFO]: Loading model checkpoint from direct path: {resume_path}")
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    if int(os.environ.get("RANK", "0")) == 0:
        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    if args_cli.frontres_eval_dr_sweep:
        if not agent_cfg.resume:
            raise ValueError("--frontres_eval_dr_sweep requires a resumed FrontRES checkpoint.")
        dr_scales = [
            float(item.strip())
            for item in args_cli.frontres_eval_dr_scales.split(",")
            if item.strip()
        ]
        output_path = os.path.join(log_dir, "frontres_dr_sweep.json")
        runner.evaluate_frontres_dr_sweep(
            dr_scales=dr_scales,
            num_iterations_per_scale=args_cli.frontres_eval_iterations_per_scale,
            output_path=output_path,
            init_at_random_ep_len=True,
        )
        env.close()
        return

    if args_cli.frontres_segment_live_update_loop_only:
        runner.run_frontres_segment_live_update_loop(init_at_random_ep_len=True)
        env.close()
        return

    if (
        args_cli.frontres_segment_live_probe_only
        or args_cli.frontres_segment_live_storage_write_only
        or args_cli.frontres_segment_live_single_update_only
    ):
        runner.run_frontres_segment_live_probe(init_at_random_ep_len=True)
        env.close()
        return

    if bool(getattr(getattr(agent_cfg, "algorithm", None), "frontres_segment_live_train_enabled", False)):
        runner.learn_frontres_segment_live(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
        env.close()
        return

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    
    # close sim app
    simulation_app.close()
