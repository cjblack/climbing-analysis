"""Interlimb timing / phase from movement-event onsets.

Quantifies *when* one limb moves relative to another — the behavioural control
for interpreting apparent bilateral neural tuning. If limbs tend to move together
(phase ≈ 0) or in strict alternation (phase ≈ 0.5), then neural responses aligned
to different limbs are confounded by co-movement, and a unit can look "bilateral"
without encoding both limbs. Measuring the interlimb phase tells you how severe
that confound is (and whether it is in-phase or anti-phase, which matters for
whether *opponent* tuning could be spurious).

For an ordered limb pair (A, B), the phase of each A onset is computed within B's
inter-onset interval that brackets it: ``(t_A - t_B_prev) / (t_B_next - t_B_prev)``
∈ [0, 1] — 0/1 = A moves at B's onset (in-phase), 0.5 = A moves mid-cycle
(anti-phase). The circular resultant length R summarises concentration (0 = no
timing relationship, 1 = perfectly phase-locked).
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from neurokinematics.io import load_csv


def circular_resultant(phases):
    """Mean resultant length R and mean phase for phases in [0, 1].

    Args:
        phases (np.ndarray): Phase values in [0, 1].

    Returns:
        tuple: ``(R, mean_phase)`` — R in [0, 1] (0 = uniform, 1 = locked), mean
        phase in [0, 1]. Both NaN when empty.
    """
    phases = np.asarray(phases, dtype=float)
    phases = phases[np.isfinite(phases)]
    if phases.size == 0:
        return float('nan'), float('nan')
    z = np.mean(np.exp(1j * 2 * np.pi * phases))
    return float(np.abs(z)), float((np.angle(z) % (2 * np.pi)) / (2 * np.pi))


def interlimb_phase(alignment, node_pairs: list | None = None, event: str = "start"):
    """Phase of each limb's movement onset within another limb's cycle.

    Args:
        alignment (pd.DataFrame | str | Path): Movement-event alignment table (or
            path to ``movement_event_alignment.csv``) with columns ``trial``,
            ``node``, ``movement_event``, ``event_times_ts``.
        node_pairs (list | None): Ordered ``(A, B)`` pairs to compute (phase of A
            within B's cycle). Defaults to all unordered limb pairs.
        event (str): Movement event used as the onset marker. Defaults to ``'start'``.

    Returns:
        dict: ``{(A, B): np.ndarray of phases in [0, 1]}`` pooled across trials.

    Example:
        >>> phases = interlimb_phase("movement_event_alignment.csv")
        >>> circular_resultant(phases[('l_forepaw', 'r_forepaw')])
    """
    if isinstance(alignment, (str, Path)):
        alignment = load_csv(alignment, method='pandas')

    df = alignment[alignment['movement_event'].astype(str) == str(event)]
    nodes = sorted(df['node'].astype(str).unique())
    if node_pairs is None:
        node_pairs = list(itertools.combinations(nodes, 2))

    out = {}
    for a, b in node_pairs:
        phases = []
        for _, g in df.groupby('trial'):
            ta = np.sort(g.loc[g['node'].astype(str) == str(a), 'event_times_ts'].astype(float).values)
            tb = np.sort(g.loc[g['node'].astype(str) == str(b), 'event_times_ts'].astype(float).values)
            if ta.size == 0 or tb.size < 2:
                continue
            for x in ta:
                idx = int(np.searchsorted(tb, x))
                if idx <= 0 or idx >= tb.size:
                    continue                      # no bracketing B onsets
                t0, t1 = tb[idx - 1], tb[idx]
                if t1 > t0:
                    phases.append((x - t0) / (t1 - t0))
        out[(a, b)] = np.asarray(phases, dtype=float)
    return out


def plot_interlimb_phase(phases, bins: int = 18, save_path: Path | str | None = None):
    """Polar histogram of interlimb phase per pair, annotated with R and n.

    Args:
        phases (dict): Output of :func:`interlimb_phase`.
        bins (int): Number of angular bins.
        save_path (Path | str | None): If given, save a ``.png`` here.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    import math
    import matplotlib.pyplot as plt

    pairs = list(phases.keys())
    n = max(len(pairs), 1)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows),
                            subplot_kw={'projection': 'polar'}, squeeze=False)
    edges = np.linspace(0, 2 * np.pi, bins + 1)
    for i, pair in enumerate(pairs):
        ax = axs[i // ncols][i % ncols]
        ph = phases[pair]
        ph = ph[np.isfinite(ph)]
        counts, _ = np.histogram(2 * np.pi * ph, bins=edges)
        ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge',
               color='#6366f1', alpha=0.8, edgecolor='white', linewidth=0.3)
        R, _ = circular_resultant(ph)
        ax.set_title(f"{pair[0]} in {pair[1]}\nR = {R:.2f}, n = {ph.size}", fontsize=9)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(['0 / in-phase', '.25', '.5\nanti-phase', '.75'], fontsize=7)
        ax.set_yticklabels([])
    for j in range(len(pairs), nrows * ncols):
        axs[j // ncols][j % ncols].axis('off')
    fig.suptitle("Interlimb phase — onset of A within B's cycle "
                 "(0/1 = in-phase, 0.5 = anti-phase; R = concentration)")
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plot_path = save_path if save_path.suffix else (save_path / "interlimb_phase.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path.as_posix(), dpi=120, bbox_inches="tight")

    return fig
