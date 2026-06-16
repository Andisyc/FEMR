# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode bookkeeping helpers for runner rollout loops."""

from __future__ import annotations

from typing import Any

import torch


def update_episode_bookkeeping(
    runner: Any,
    *,
    infos: dict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    intrinsic_rewards: torch.Tensor | None,
    ep_infos: list,
    rewbuffer: Any,
    lenbuffer: Any,
    rewbuffer_gmt: Any,
    lenbuffer_gmt: Any,
    lenbuffer_gmt_base: Any,
    lenbuffer_gmt_base_frontier: Any,
    erewbuffer: Any,
    irewbuffer: Any,
    cur_reward_sum: torch.Tensor,
    cur_episode_length: torch.Tensor,
    cur_reward_sum_gmt: torch.Tensor | None,
    cur_ereward_sum: torch.Tensor | None,
    cur_ireward_sum: torch.Tensor | None,
    is_frontres: bool,
    n_train: int,
    n_candidate: int,
    n_base: int,
    n_clean: int,
    r_candidate_gmt: torch.Tensor | None,
    r_raw_gmt: torch.Tensor | None,
    r_clean_gmt: torch.Tensor | None,
) -> None:
    """Update episode buffers after a rollout step has been stored."""
    if runner.log_dir is None:
        return

    if "episode" in infos:
        ep_infos.append(infos["episode"])
    elif "log" in infos:
        ep_infos.append(infos["log"])

    has_rnd = hasattr(runner.alg, "rnd") and runner.alg.rnd
    if has_rnd:
        if (
            cur_ereward_sum is None
            or cur_ireward_sum is None
            or intrinsic_rewards is None
            or erewbuffer is None
            or irewbuffer is None
        ):
            raise RuntimeError("RND bookkeeping requires extrinsic/intrinsic reward buffers.")
        cur_ereward_sum += rewards
        cur_ireward_sum += intrinsic_rewards
        cur_reward_sum += rewards + intrinsic_rewards
    else:
        cur_reward_sum += rewards

    if is_frontres:
        if cur_reward_sum_gmt is None or r_raw_gmt is None:
            raise RuntimeError("FrontRES bookkeeping requires GMT reward buffers.")
        if n_candidate > 0 and r_candidate_gmt is not None:
            cur_reward_sum_gmt[:n_candidate] += r_candidate_gmt
        gmt_base_local = n_candidate
        cur_reward_sum_gmt[gmt_base_local:gmt_base_local + n_base] += r_raw_gmt
        if n_clean > 0:
            if r_clean_gmt is None:
                raise RuntimeError("FrontRES clean-GMT bookkeeping requires r_clean_gmt.")
            gmt_clean_local = gmt_base_local + n_base
            cur_reward_sum_gmt[gmt_clean_local:gmt_clean_local + n_clean] += r_clean_gmt

    cur_episode_length += 1

    new_ids = (dones > 0).nonzero(as_tuple=False)
    if is_frontres and len(new_ids) > 0:
        if cur_reward_sum_gmt is None:
            raise RuntimeError("FrontRES done bookkeeping requires cur_reward_sum_gmt.")
        env_idx = new_ids[:, 0]
        fr_done = new_ids[env_idx < n_train]
        gmt_done = new_ids[env_idx >= n_train]
        if len(fr_done) > 0:
            rewbuffer.extend(cur_reward_sum[fr_done][:, 0].cpu().numpy().tolist())
            lenbuffer.extend(cur_episode_length[fr_done][:, 0].cpu().numpy().tolist())
        if len(gmt_done) > 0:
            gmt_local = gmt_done[:, 0] - n_train
            rewbuffer_gmt.extend(cur_reward_sum_gmt[gmt_local].cpu().numpy().tolist())
            lenbuffer_gmt.extend(cur_episode_length[gmt_done][:, 0].cpu().numpy().tolist())
            base_mask = (gmt_local >= n_candidate) & (gmt_local < n_candidate + n_base)
            if bool(base_mask.any().item()):
                lenbuffer_gmt_base.extend(
                    cur_episode_length[gmt_done[base_mask]][:, 0].cpu().numpy().tolist()
                )
                mix_class = getattr(runner, "_frontres_dr_mix_class_train", None)
                if mix_class is not None and n_base > 0:
                    base_local = gmt_local[base_mask] - n_candidate
                    base_local = base_local.to(device=runner.device, dtype=torch.long)
                    valid = (base_local >= 0) & (base_local < int(mix_class.numel()))
                    if bool(valid.any().item()):
                        frontier_mask = mix_class[base_local[valid]] == 1
                        if bool(frontier_mask.any().item()):
                            frontier_done = gmt_done[base_mask][valid][frontier_mask]
                            lenbuffer_gmt_base_frontier.extend(
                                cur_episode_length[frontier_done][:, 0].cpu().numpy().tolist()
                            )
            cur_reward_sum_gmt[gmt_local] = 0
    elif len(new_ids) > 0:
        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())

    cur_reward_sum[new_ids] = 0
    cur_episode_length[new_ids] = 0

    if has_rnd:
        if cur_ereward_sum is None or cur_ireward_sum is None:
            raise RuntimeError("RND done bookkeeping requires reward buffers.")
        erewbuffer.extend(cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
        irewbuffer.extend(cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
        cur_ereward_sum[new_ids] = 0
        cur_ireward_sum[new_ids] = 0
