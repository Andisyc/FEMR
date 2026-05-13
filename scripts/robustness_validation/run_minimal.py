"""Minimal validation smoke test: 1 env, 1 epsilon, 1 push, 1 trial."""
import argparse
import faulthandler
import os
import sys
from pathlib import Path


def log(message: str) -> None:
    print(message, flush=True)


def _sanitize_python_path_for_isaac() -> None:
    """Avoid loading binary packages from the user's site-packages into Isaac."""

    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    try:
        import site

        user_site = site.getusersitepackages()
    except Exception:
        user_site = None

    def _is_user_site_path(path: str) -> bool:
        if not path:
            return False
        if isinstance(user_site, str) and path == user_site:
            return True
        return "/.local/lib/python" in path and "site-packages" in path

    sys.path[:] = [path for path in sys.path if not _is_user_site_path(path)]

    if "numpy" in sys.modules:
        del sys.modules["numpy"]


_sanitize_python_path_for_isaac()

from isaaclab.app import AppLauncher

TASK = "Tracking-Flat-G1-Wo-State-Estimation-v0"

parser = argparse.ArgumentParser(description="Minimal robustness validation smoke test.")
parser.add_argument("--motion",     type=str, required=True)
parser.add_argument("--checkpoint", "--resume_path", dest="checkpoint", type=str, required=True)
parser.add_argument("--task",       type=str, default=TASK)
parser.add_argument("--num_envs",   type=int, default=1)
parser.add_argument("--video",      action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument(
    "--startup_timeout",
    type=int,
    default=30,
    help="Dump Python stack traces every N seconds while diagnosing startup hangs. Use 0 to disable.",
)
parser.add_argument(
    "--keep_events",
    action="store_true",
    default=False,
    help="Keep event/curriculum managers instead of disabling them like play.py.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.startup_timeout > 0:
    faulthandler.enable()
    faulthandler.dump_traceback_later(args_cli.startup_timeout, repeat=True)


def _resolve_motion_path(path: str) -> str:
    motion_path = Path(path).expanduser()
    if motion_path.is_file():
        return str(motion_path)
    if motion_path.is_dir():
        candidates = sorted(motion_path.rglob("*.npz"))
        if candidates:
            selected = candidates[0]
            log(f"[SMOKE] --motion is a directory; using first .npz: {selected}")
            return str(selected)
    parser.error(
        f"--motion must be a .npz file for task {args_cli.task!r}, or a directory containing .npz files: {path}"
    )


args_cli.motion = _resolve_motion_path(args_cli.motion)

if args_cli.video:
    args_cli.enable_cameras = True

if (
    sys.platform.startswith("linux")
    and not args_cli.headless
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    log("[SMOKE] No display detected; forcing --headless for Isaac Sim startup.")
    args_cli.headless = True

# This script does not use Hydra.  Leaving unknown CLI fragments in sys.argv can
# make Kit consume arguments that were intended for other launch paths.
if hydra_args:
    log(f"[SMOKE] Ignoring non-AppLauncher arguments: {hydra_args}")
sys.argv = [sys.argv[0]]

log(f"[SMOKE] Launching Isaac Sim via AppLauncher: device={args_cli.device}, headless={args_cli.headless}")
app_launcher = AppLauncher(args_cli)
log("[SMOKE] AppLauncher constructed; retrieving simulation_app...")
simulation_app = app_launcher.app
log("[SMOKE] simulation_app is ready.")

# ── After Isaac Sim is running ──────────────────────────────────────────
import torch
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import whole_body_tracking.tasks  # noqa

log("[SMOKE] Isaac Sim running, importing modules...")

# ── Minimal env build ───────────────────────────────────────────────────
task_entry = gym.envs.registry[args_cli.task]
env_cfg_cls = task_entry.kwargs["env_cfg_entry_point"]

@configclass
class _NoOpCfg:
    pass


# Monkey-patch: replace None managers with empty configs.
_orig = getattr(env_cfg_cls, '__post_init__', None)

def _safe_post_init(self_):
    if _orig is not None:
        _orig(self_)
    if getattr(self_, 'events', None) is None:
        self_.events = _NoOpCfg()
    if getattr(self_, 'curriculum', None) is None:
        self_.curriculum = _NoOpCfg()

env_cfg_cls.__post_init__ = _safe_post_init

env_cfg = env_cfg_cls()
if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
    env_cfg.sim.device = args_cli.device
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.commands.motion.motion = args_cli.motion   # MotionCommandCfg 的正确字段名
if hasattr(env_cfg.commands.motion, "motion_file"):
    env_cfg.commands.motion.motion_file = args_cli.motion
if hasattr(env_cfg.commands.motion, "start_from_beginning"):
    env_cfg.commands.motion.start_from_beginning = True
if hasattr(env_cfg.commands.motion, "start_frame"):
    env_cfg.commands.motion.start_frame = 0

# Disable timeout
if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
    env_cfg.terminations.time_out = None

# Disable event/curriculum terms without setting the manager configs to None.
# Isaac Lab registers manager callbacks before sim.reset(), and callbacks assume
# cfg has a __dict__ even when there are no terms.
if not args_cli.keep_events:
    if hasattr(env_cfg, "events"):
        env_cfg.events = _NoOpCfg()
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = _NoOpCfg()

# Headless smoke tests should not register debug-visualization callbacks.  The
# contact sensor callback can block inside PhysX tensor reads during sim.reset().
if hasattr(env_cfg.commands.motion, "debug_vis"):
    env_cfg.commands.motion.debug_vis = False
if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "contact_forces"):
    env_cfg.scene.contact_forces.debug_vis = False

# Zero init randomisation
motion_cfg = getattr(env_cfg.commands, "motion", None)
if motion_cfg is not None:
    zero = {"x": (0., 0.), "y": (0., 0.), "z": (0., 0.),
            "roll": (0., 0.), "pitch": (0., 0.), "yaw": (0., 0.)}
    for attr in ("pose_range", "velocity_range"):
        if hasattr(motion_cfg, attr):
            setattr(motion_cfg, attr, zero)

log("[SMOKE] Creating env...")
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
log("[SMOKE] gym.make returned.")
if args_cli.video:
    video_kwargs = {
        "video_folder": os.path.join(os.path.dirname(args_cli.checkpoint), "videos", "run_minimal"),
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)
env = RslRlVecEnvWrapper(env)
log(f"[SMOKE] Env created: {env.num_envs} envs")

# ── Load GMT ────────────────────────────────────────────────────────────
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

agent_cfg = G1FlatPPORunnerCfg()
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device)
runner._move_normalizer_to_device(args_cli.device)
runner.load(args_cli.checkpoint, load_optimizer=False, load_critic=False)
log("[SMOKE] GMT loaded")

# ── Run one trial ───────────────────────────────────────────────────────
policy = runner.get_inference_policy(device=args_cli.device)
obs, _ = env.get_observations()

# Disable motion perturbations for clean baseline
env_unwrapped = env.unwrapped
cmd = env_unwrapped.command_manager._terms.get("motion")
if cmd is not None and hasattr(cmd, 'perturber'):
    cfg = cmd.perturber.cfg
    cfg.float_prob = 0.0
    cfg.sink_prob = 0.0
    cfg.root_tilt_prob = 0.0
    cfg.joint_noise_prob = 0.0

log("[SMOKE] Running 300 steps...")
for step in range(300):
    with torch.inference_mode():
        actions = policy(obs)
    obs, _, dones, _ = env.step(actions)
    if step % 50 == 0:
        log(f"   step {step:3d}: done={dones.detach().cpu().tolist()}")

log("[SMOKE] OK — 300 steps completed without crash")
env.close()
simulation_app.close()
faulthandler.cancel_dump_traceback_later()
log("[SMOKE] Done.")
