from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "source" / "rsl_rl" / "rsl_rl" / "frontres" / "frontres_policy_quality_manifest.py"
spec = importlib.util.spec_from_file_location("frontres_policy_quality_manifest", MODULE_PATH)
manifest_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = manifest_module
spec.loader.exec_module(manifest_module)

FrontRESPolicyQualityManifest = manifest_module.FrontRESPolicyQualityManifest
FrontRESPolicyQualityManifestItem = manifest_module.FrontRESPolicyQualityManifestItem
FrontRESPolicyQualityRouteIdentity = manifest_module.FrontRESPolicyQualityRouteIdentity
FrontRESPolicyQualityStateIdentity = manifest_module.FrontRESPolicyQualityStateIdentity
FrontRESV017PolicyQualityManifest = manifest_module.FrontRESV017PolicyQualityManifest


def _item(*, item_id: str = "motion-a-k8", **changes: object) -> FrontRESPolicyQualityManifestItem:
    values: dict[str, object] = {
        "item_id": item_id,
        "motion_id": "KIT/314/parkour08.npz",
        "start_frame": 48,
        "perturbation_family": "local_rp",
        "perturbation_parameters": (("pitch_rad", -0.08), ("roll_rad", 0.12)),
        "effective_horizon_k": 8,
        "seed": 17,
    }
    values.update(changes)
    return FrontRESPolicyQualityManifestItem(**values)


def _manifest(items: tuple[FrontRESPolicyQualityManifestItem, ...]) -> FrontRESPolicyQualityManifest:
    return FrontRESPolicyQualityManifest(
        environment_revision="isaaclab-mosaic-20260717",
        config_revision="frontres-stage3-rp-v1",
        evaluator_version="policy-quality-v1",
        items=items,
    )


def test_hand_checkable_signature_and_round_trip() -> None:
    manifest = _manifest((_item(),))
    loaded = FrontRESPolicyQualityManifest.from_json(manifest.to_json())
    assert loaded == manifest
    assert loaded.comparison_signature == manifest.comparison_signature
    assert loaded.items[0].comparison_signature == manifest.items[0].comparison_signature
    assert len(manifest.comparison_signature) == 64
    print(
        "[quality manifest trace] "
        f"item={manifest.items[0].comparison_signature} manifest={manifest.comparison_signature}"
    )


def test_semantic_identity_is_order_independent_but_control_sensitive() -> None:
    first = _item(item_id="a")
    second = _item(item_id="b", motion_id="CMU/29/29_04.npz", seed=23)
    assert _manifest((first, second)).comparison_signature == _manifest((second, first)).comparison_signature

    mutations = (
        replace(first, motion_id="other.npz"),
        replace(first, start_frame=49),
        replace(first, perturbation_parameters=(("pitch_rad", -0.08), ("roll_rad", 0.13))),
        replace(first, effective_horizon_k=16),
        replace(first, seed=18),
    )
    assert all(item.comparison_signature != first.comparison_signature for item in mutations)
    manifest = _manifest((first, second))
    assert replace(manifest, environment_revision="other-env").comparison_signature != manifest.comparison_signature
    assert replace(manifest, config_revision="other-config").comparison_signature != manifest.comparison_signature
    assert replace(manifest, evaluator_version="policy-quality-v2").comparison_signature != manifest.comparison_signature
    assert replace(first, item_id="display-name-only").comparison_signature == first.comparison_signature


def test_manifest_is_immutable_and_rejects_missing_duplicate_or_extra_fields() -> None:
    manifest = _manifest((_item(),))
    try:
        manifest.items = ()  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("manifest must be immutable")

    duplicate = replace(_item(item_id="duplicate"))
    try:
        _manifest((duplicate, duplicate))
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate comparison rows must fail closed")

    try:
        _manifest((duplicate, replace(duplicate, item_id="same-question-other-label")))
    except ValueError as exc:
        assert "duplicate comparison identity" in str(exc)
    else:
        raise AssertionError("semantic duplicates with different labels must fail closed")

    payload = manifest.to_dict()
    del payload["items"][0]["motion_id"]
    try:
        FrontRESPolicyQualityManifest.from_dict(payload)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing identity field must fail closed")

    payload = manifest.to_dict()
    payload["checkpoint"] = "model_700.pt"
    try:
        FrontRESPolicyQualityManifest.from_dict(payload)
    except ValueError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("checkpoint must not enter manifest identity")

    payload = manifest.to_dict()
    payload["sampler_state"] = {"cursor": 99}
    try:
        FrontRESPolicyQualityManifest.from_dict(payload)
    except ValueError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("sampler state must not enter manifest identity")


def test_checkpoint_is_route_metadata_not_comparison_identity() -> None:
    item = _item()
    state = FrontRESPolicyQualityStateIdentity(
        comparison_signature=item.comparison_signature,
        initial_state_hash="a" * 64,
    )
    model_200 = FrontRESPolicyQualityRouteIdentity(
        route="hsl",
        checkpoint_identity="model_200:sha256-a",
        state=state,
    )
    model_700 = FrontRESPolicyQualityRouteIdentity(
        route="policy",
        checkpoint_identity="model_700:sha256-b",
        state=state,
    )
    assert model_200.comparison_signature == model_700.comparison_signature == item.comparison_signature
    assert model_200.route_signature != model_700.route_signature


def test_v017_k16_m3_manifest_has_strict_active_identity() -> None:
    path = ROOT / "note" / "testing" / "manifests" / "frontres_v017_policy_quality_k16_v1.json"
    manifest = FrontRESV017PolicyQualityManifest.from_json(path.read_text(encoding="utf-8"))
    assert manifest.method_contract_id == "FRS-METHOD-v017"
    assert manifest.training_contract_id == "FRS-TRAIN-v015"
    assert manifest.gain_contract_id == "FRS-GAIN-v007"
    assert manifest.ppo_contract_id == "FRS-PPO-v005"
    assert manifest.evaluation_contract_id == "FRS-EVAL-v004"
    assert manifest.checkpoint_format == "frontres-v017-checkpoint-v10"
    assert (manifest.horizon_k, manifest.attempts_per_segment, manifest.segments_per_transaction) == (16, 3, 2)
    assert len(manifest.items) == 8
    assert len({(item.motion_id, item.start_frame) for item in manifest.items}) == 8
    assert all(item.effective_horizon_k == 16 for item in manifest.items)
    assert FrontRESV017PolicyQualityManifest.from_json(manifest.to_json()) == manifest

    payload = manifest.to_dict()
    payload["gain_contract_id"] = "FRS-GAIN-v006"
    try:
        FrontRESV017PolicyQualityManifest.from_dict(payload)
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:
        raise AssertionError("v017 manifest must reject legacy Gain identity")

    payload = manifest.to_dict()
    payload["items"][0]["effective_horizon_k"] = 8
    try:
        FrontRESV017PolicyQualityManifest.from_dict(payload)
    except ValueError as exc:
        assert "K16" in str(exc)
    else:
        raise AssertionError("v017 manifest must reject mixed-K items")


if __name__ == "__main__":
    test_hand_checkable_signature_and_round_trip()
    test_semantic_identity_is_order_independent_but_control_sensitive()
    test_manifest_is_immutable_and_rejects_missing_duplicate_or_extra_fields()
    test_checkpoint_is_route_metadata_not_comparison_identity()
    test_v017_k16_m3_manifest_has_strict_active_identity()
    print("PASS: immutable policy-quality manifest and comparison signatures are closed.")
