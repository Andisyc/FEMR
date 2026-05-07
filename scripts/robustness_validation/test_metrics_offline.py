"""
Offline unit test for results_io and plot_results (no Isaac Sim needed).
Run with: python test_metrics_offline.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import tempfile
from results_io import ResultsStore, TrialResult
from plot_results import load_and_plot, load_and_plot_multi


def make_fake_store() -> ResultsStore:
    meta = {
        "motion_file":     "test_motion.npz",
        "checkpoint":      "model.pt",
        "epsilon_values":  [0.0, 0.01, 0.02, 0.05, 0.10, 0.20],
        "push_velocities": [0.5, 1.0, 2.0, 3.0],
        "n_trials":        20,
        "settle_steps":    100,
        "observe_steps":   200,
        "ou_tau":          0.5,
    }
    store = ResultsStore(meta)
    rng = np.random.default_rng(42)

    eps_vals  = meta["epsilon_values"]
    pvel_vals = meta["push_velocities"]

    for ei, eps in enumerate(eps_vals):
        for pi, pvel in enumerate(pvel_vals):
            # Recovery rate decreases with epsilon and push velocity
            base_rate = max(0.0, 1.0 - eps * 4.0 - pvel * 0.1)
            for ti in range(meta["n_trials"]):
                success = rng.random() < base_rate
                # ZMP margin: worse (smaller) with higher epsilon
                settle_zmp = (rng.normal(0.06 - eps * 0.3, 0.01, 100)
                              .clip(-0.05, 0.15).tolist())
                post_zmp   = (rng.normal(0.04 - eps * 0.3, 0.02, 60)
                              .clip(-0.1, 0.15).tolist())
                store.add(ei, pi, ti, TrialResult(
                    success=bool(success),
                    fallen_before_push=False,
                    T_push_step=int(rng.integers(0, 40)),
                    zmp_margins_settle=settle_zmp,
                    zmp_margins_post=post_zmp,
                    push_dir=[float(rng.random()), float(rng.random()), 0.0],
                ))
    return store


def main():
    store = make_fake_store()

    with tempfile.TemporaryDirectory() as base:

        # ── Single-run path ──────────────────────────────────────────────────
        run1 = os.path.join(base, "run1")
        store.save(run1)

        store2 = ResultsStore.load(run1)
        assert len(store2._data) == len(store._data), "Reload mismatch"
        print("[test_metrics_offline] Save/load: OK")

        summary = store2.to_summary()
        rate_clean = summary[0][0]["recovery_rate"]
        rate_noisy = summary[5][3]["recovery_rate"]
        assert rate_clean > rate_noisy, (
            f"Expected clean > noisy: {rate_clean:.2f} vs {rate_noisy:.2f}")
        print(f"[test_metrics_offline] Summary: clean={rate_clean:.2f}, "
              f"noisy={rate_noisy:.2f} — OK")

        out_single = os.path.join(base, "figures_single")
        load_and_plot(run1, out_single)
        figs = os.listdir(out_single)
        for name in ("fig1", "fig2", "fig3", "fig4"):
            assert any(name in f for f in figs), f"{name} missing in single-run"
        print(f"[test_metrics_offline] Single-run figures: {sorted(figs)} — OK")

        # ── Multi-run path ───────────────────────────────────────────────────
        # Create 3 "different motion" runs from the same fake store
        run2 = os.path.join(base, "run2")
        run3 = os.path.join(base, "run3")
        store.save(run2)
        store.save(run3)

        out_multi = os.path.join(base, "figures_multi")
        load_and_plot_multi(
            [run1, run2, run3],
            motion_names=["walk", "dance", "squat"],
            output_dir=out_multi,
        )
        figs_multi = os.listdir(out_multi)
        for name in ("fig1", "fig2", "fig3", "fig4"):
            assert any(name in f for f in figs_multi), f"{name} missing in multi-run"
        print(f"[test_metrics_offline] Multi-run figures: {sorted(figs_multi)} — OK")

        # Verify merge_summaries produces expected keys
        named = [(n, ResultsStore.load(d).to_summary())
                 for n, d in [("walk", run1), ("dance", run2), ("squat", run3)]]
        merged = ResultsStore.merge_summaries(named)
        assert merged[0][0]["motion_names"] == ["walk", "dance", "squat"]
        assert len(merged[0][0]["rates_per_motion"]) == 3
        assert not np.isnan(merged[0][0]["mean_rate"])
        print("[test_metrics_offline] merge_summaries: OK")

    print("\n[test_metrics_offline] All tests passed.")


if __name__ == "__main__":
    main()
