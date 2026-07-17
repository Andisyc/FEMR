"""Contracts for the offline Stage 2 HSL magnitude audit."""

import importlib.util
from pathlib import Path
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "source/rsl_rl/rsl_rl/frontres/frontres_policy_quality_hsl_magnitude_audit.py"
SPEC = importlib.util.spec_from_file_location("frontres_policy_quality_hsl_magnitude_audit_contract_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
supervised_component_audit = MODULE.supervised_component_audit
checkpoint_lineage_audit = MODULE.checkpoint_lineage_audit


def main() -> None:
    target = torch.tensor([[0.0, 0.0, 0.0, 0.02, 0.0, 0.0]])
    over = torch.tensor([[0.0, 0.0, 0.0, 0.20, 0.0, 0.0]])
    closer = torch.tensor([[0.0, 0.0, 0.0, 0.04, 0.0, 0.0]])
    weight = torch.ones(1)
    harm = torch.ones(1)
    over_audit = supervised_component_audit(over, target, weight, harm)
    closer_audit = supervised_component_audit(closer, target, weight, harm)

    assert over_audit["action_target_norm_ratio"]["median"] == 10.0
    assert closer_audit["components"]["magnitude"]["weighted_loss"] < over_audit["components"]["magnitude"]["weighted_loss"]
    assert closer_audit["components"]["over"]["weighted_loss"] < over_audit["components"]["over"]["weighted_loss"]
    assert over_audit["components"]["magnitude"]["proposal_grad_l2"] > 0.0
    assert over_audit["components"]["over"]["proposal_grad_l2"] > 0.0
    assert over_audit["components"]["direction_rpy"]["weighted_loss"] == 0.0

    near_zero = torch.tensor([[1.0e-7, 1.0e-7, 0.0, 0.02, 0.0, 0.0]])
    mixed_target = torch.tensor([[0.01, 0.0, 0.0, 0.02, 0.0, 0.0]])
    singular = supervised_component_audit(near_zero, mixed_target, weight, harm)
    assert singular["components"]["direction_pos"]["proposal_grad_l2"] > 1.0
    assert singular["gradient_competition"]["largest_component"] == "direction_pos"

    with tempfile.TemporaryDirectory() as tmp:
        incomplete_path = Path(tmp) / "incomplete.pt"
        torch.save({"iter": 200, "model_state_dict": {}}, incomplete_path)
        incomplete = checkpoint_lineage_audit(str(incomplete_path), ["server/model_200.pt"])
        assert incomplete["status"] == "lineage_incomplete"
        assert incomplete["source_checkpoint_identity"] is None
        assert "supervised_magnitude_loss_weight" in incomplete["missing_supervised_config_keys"]
    print("PASS: HSL magnitude audit separates scale losses and gradients.")


if __name__ == "__main__":
    main()
