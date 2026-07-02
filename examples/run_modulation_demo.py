"""Demo: event_modulation on a synthetic session, with a Fig-3 style z-heatmap.

Builds a synthetic movement-aligned rasters_df (same schema as
get_movement_aligned_rasters), injects a few biologically-plausible modulation
patterns, runs event_modulation, prints the significant cells, and saves a
population z-scored heatmap (units x time, one column per epoch).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from neurokinematics.ephys.spikes.modulation import event_modulation

# reuse the synthetic raster builder from the test suite
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "unit"))
from test_modulation import make_rasters_df  # noqa: E402


def main():
    units = list(range(8))
    nodes = ["l_forepaw", "r_forepaw", "l_hindpaw", "r_hindpaw"]
    epochs = ["start", "max", "end"]
    # injected ground truth: rate added (+) or suppressed (-) in [0, 0.2] s
    mod = {
        (0, "l_forepaw", "start"): 50.0,   # ipsi forepaw, movement initiation
        (2, "r_forepaw", "max"):   45.0,   # peak-velocity tuned
        (4, "l_hindpaw", "end"):  -18.0,   # suppressed at cessation
        (6, "l_forepaw", "start"): 40.0,   # unit 6 = BILATERAL pattern:
        (6, "l_hindpaw", "start"): 35.0,   #   up for ipsi (left) limbs ...
        (6, "r_forepaw", "start"): -15.0,  #   ... down for contra (right) limbs
        (6, "r_hindpaw", "start"): -15.0,
    }
    df = make_rasters_df(units=units, nodes=nodes, epochs=epochs,
                         n_events=40, baseline_rate=15.0, seed=7, modulated=mod)
    ds = event_modulation(df, n_shuffle=1000, seed=0)

    print("\n=== significant (FDR<0.05) cells ===")
    sig = ds["significant"]
    for ui, ni, ei in np.argwhere(sig.values):
        c = ds.isel(unit=ui, node=ni, epoch=ei)
        print(f"unit {int(ds.unit[ui]):>2}  {str(ds.node[ni].values):11s}  "
              f"{str(ds.epoch[ei].values):5s}  mod={c['modulation'].item():+6.1f} Hz  "
              f"z={c['response_z'].item():+5.2f}  p={c['p_value'].item():.4f}  "
              f"p_fdr={c['p_fdr'].item():.4f}")
    print(f"\ntotal significant: {int(sig.sum())} / {sig.size} cells tested")

    node = "l_forepaw"
    t = ds["time_bin"].values
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, ep in zip(axes, epochs):
        Z = ds["psth_z"].sel(node=node, epoch=ep).values
        im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4,
                       extent=[t[0], t[-1], len(units) - 0.5, -0.5],
                       interpolation="nearest")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_title(ep)
        ax.set_xlabel("time from event (s)")
        for u in np.where(ds["significant"].sel(node=node, epoch=ep).values)[0]:
            ax.text(t[-1] * 0.9, u, "*", va="center", ha="center", fontsize=14)
    axes[0].set_ylabel("unit")
    fig.suptitle(f"Event-aligned z-scored firing — {node}  (* = FDR<0.05)")
    fig.colorbar(im, ax=axes, label="z (vs baseline)", shrink=0.8)
    out = Path(__file__).resolve().parent / "fig3_modulation_demo.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("saved:", out)


if __name__ == "__main__":
    main()
