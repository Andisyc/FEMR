#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE_CFG = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
G1_CFG = ROOT / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py"


def _index_after(text: str, needle: str, after: int = 0) -> int:
    index = text.find(needle, after)
    assert index >= 0, f"missing expected text: {needle}"
    return index


def test_frontres_balance_obs_is_optional_tail_term() -> None:
    base_text = BASE_CFG.read_text()
    g1_text = G1_CFG.read_text()

    base_pos = _index_after(base_text, "anchor_root_pos_error_w: ObsTerm | None = None")
    base_rpy = _index_after(base_text, "anchor_root_rpy_error_w: ObsTerm | None = None", base_pos)
    base_balance = _index_after(base_text, "frontres_balance_context: ObsTerm | None = None", base_rpy)
    assert base_pos < base_rpy < base_balance

    g1_pos = _index_after(g1_text, "self.observations.policy.anchor_root_pos_error_w = ObsTerm(")
    g1_rpy = _index_after(g1_text, "self.observations.policy.anchor_root_rpy_error_w = ObsTerm(", g1_pos)
    g1_balance = _index_after(g1_text, "self.observations.policy.frontres_balance_context = ObsTerm(", g1_rpy)
    g1_termination = _index_after(g1_text, "self.terminations.ee_body_pos.params", g1_balance)
    assert g1_pos < g1_rpy < g1_balance < g1_termination

    block = g1_text[g1_balance:g1_termination]
    assert "func=mdp.frontres_balance_context_proxy" in block
    assert 'params={"command_name": "motion"}' in block
    assert "870 dims total" in g1_text
    assert "[30:100]" in g1_text
    assert "[100:870]" in g1_text


if __name__ == "__main__":
    test_frontres_balance_obs_is_optional_tail_term()
    print("frontres_balance_obs_cfg_contract: ok")
