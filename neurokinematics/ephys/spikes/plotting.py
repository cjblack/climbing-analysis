"""Plotting utilities for spike data.

Utilities for visualising spike-sorting outputs and features.
Provides a lightweight abstraction layer over spikeinterface plotting tools along with project-specific plotting fucntions to simplify plotting and saving figures.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spikeinterface.widgets as sw
from neurokinematics.ephys.io import *



# Simple plots for spike data

def plot_waveforms(analyzer, unit_ids: list, max_spikes: int = 100, save_path: Path | str | None = None):
    """Plots individual and average waveforms of specified units across identified channels.

    Args:
        analyzer (SortingAnalyzer): Spike sorting analyzer from spikeinterface, can either be used from running the sort function or loading directly from a save.
        unit_ids (list): List of unit ids to plot.
        max_spikes (int, optional): Maximum number of single spike waveforms to plot, best to set this number low, especially when plotting multiple units. Defaults to 100.
        save_path (Path | str | None, optional): Determines whether plot is saved and to where. Figure will be saved in the save_path directory as a '.png'. Defaults to None.

    Example:
        >>> plot_waveforms(
        ...     analyzer = analyzer,
        ...     unit_ids = [5, 10, 15],
        ...     max_spikes = 100,
        ...     save_path = "path/to/outputs"     
        ...     )
    """
    
    # lazy correction if plotting one unit
    if not isinstance(unit_ids, list):
        unit_ids = [unit_ids]

    # plot unit waveforms using spikewidget function
    sw.plot_unit_waveforms(analyzer, unit_ids=unit_ids, max_spikes_per_unit=max_spikes)
    plt.suptitle('Unit Waveforms')
    plt.tight_layout()

    if save_path:
        plots_dir = Path(save_path) / 'unit_plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / 'unit_waveforms.png'
        plt.savefig(plot_path.as_posix()) # save figure to analyzer path

    plt.show()


def plot_autocorrelogram(sorter, unit_ids: list, save_path: Path | str | None = None):
    """Plots autocorrelogram for specified units.

    Args:
        sorter (SortingExtractor): Spikeinterface Sorting Extractor object. Get from either running sort, or loading from previous sorting.
        unit_ids (list): List of unit ids to plot.
        save_path (Path | str | None, optional): Determines whether plot is saved and to where. Figure will be saved in the save_path directory as a '.png'. Defaults to None.
    
    Example:
        >>> plot_autocorrelogram(
        ...     sorter = sorter,
        ...     unit_ids = [5, 10, 15],
        ...     save_path = "path/to/outputs"
        ...     )
    """
    
    # lazy correction if plotting one unit
    if not isinstance(unit_ids,list):
        unit_ids = [unit_ids]

    # plot autocorrelograms using spikewidget function
    w = sw.plot_autocorrelograms(sorter, unit_ids=unit_ids)
    plt.suptitle('Unit Autocorrelograms')
    plt.tight_layout()

    if save_path:
        plots_dir = Path(save_path) / 'unit_plots'#Path(sorter.get_annotation('phy_folder')).parent.parent / 'unit_plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / 'unit_autocorrelograms.png'
        plt.savefig(plot_path.as_posix()) # save figure to analyzer path

    plt.show()


def plot_movement_psth(rasters_df: pd.DataFrame, unit_ids: list, movement_plot_params: dict | None = None, save_path: Path | str | None = None):
    """Plot psth with respect to movement events.

    Args:
        rasters_df (pd.DataFrame): Dataframe containing spike rasters aligned to movement.
        unit_ids (list): List of unit ids to plot.
        movement_plot_params (dict | None, optional): Dictionary containing parameters for plotting requires:
            {
            'node': str, body part (node). This will be based on the nodes using during markerless pose estimation.
            'movement_event': str, type of movement (e.g. 'start', 'end', 'max'). This will be based on the movement events you extract.
            'cmap': str, matplotlib colormap to use.
            'bin_size': Bin size in seconds for psth (default is 0.05).
            }
        
            Defaults to None, which defaults to plotting rasters with respect to the first rows node and event type, in black.

        save_path (Path | str | None, optional): Determines whether plot is saved and to where. Figure will be saved in the save_path directory as a '.png'. Defaults to None.
    
    Example:
        >>> plot_movement_psth(
        ...     rasters_df = rasters_df,
        ...     unit_ids = [5, 10, 15],
        ...     movement_plot_params = {
        ...         'node': 'r_hindpaw',
        ...         'movement_event': 'end',
        ...         'cmap': 'winter'
        ...         'bin_size': 0.05
        ...         },
        ...     save_path = "path/to/outputs"
        ...     )
    """
    # lazy correction if plotting one unit
    if not isinstance(unit_ids,list):
        unit_ids = [unit_ids]
    
    # get movement plot params
    if movement_plot_params:
        # dictionary extraction if provided
        node = movement_plot_params['node']
        movement_event = movement_plot_params['movement_event']
        bin_size = movement_plot_params['bin_size']
        mpl_cmap = movement_plot_params['cmap']
    else:
        # if no movement_plot_params given then defaults to first node and movement event
        node = rasters_df['node'].unique()[0]
        movement_event = rasters_df['movement_event'].unique()[0]
        bin_size = 0.05
        mpl_cmap = 'default'

    n = len(unit_ids)
    ncols = min(5, n)
    n_unit_rows = math.ceil(n / ncols)
    if mpl_cmap == 'default':
        cmap = lambda i: 'black'
    else:
        cmap = plt.get_cmap(mpl_cmap, n)

    fig, axes = plt.subplots(
        n_unit_rows * 2,
        ncols,
        figsize=(3 * ncols, 4 * n_unit_rows),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1] * n_unit_rows},
    )

    axes = np.array(axes).reshape(n_unit_rows * 2, ncols)
    
    for i, uid in enumerate(unit_ids):
        rasters = rasters_df.query("unit_id==@uid & node==@node & movement_event==@movement_event")
        #raster_index = rasters.index[0] # correct for starting position in row of raster
        unit_row = i // ncols
        col = i % ncols
        raster_ax = axes[unit_row * 2, col]
        psth_ax = axes[unit_row * 2 + 1, col]
        #ax = axes[row, col]
        if rasters.empty:
            raster_ax.set_title(f"Unit id: {uid} (no data)")
            psth_ax.axis("off")
            continue
        all_spikes = []
        for trial_idx, (_, trial_row) in enumerate(rasters.iterrows()):
            #pos_ = ii-raster_index
            spks = np.asarray(trial_row['spike_raster'])
            all_spikes.extend(spks)
            raster_ax.vlines(
                spks,
                trial_idx,
                trial_idx+1,
                color='black',
                lw=1
            )
            #ax.vlines(spks, pos_+0, pos_+1, color=cmap(i), lw=1)
        all_spikes = np.asarray(all_spikes)
        if len(all_spikes) > 0:
            #t_min = all_spikes.min()
            #t_max = all_spikes.max()
            t_min = -0.5
            t_max = 0.5
            bins = np.arange(t_min, t_max + bin_size, bin_size)
            counts, edges = np.histogram(all_spikes, bins = bins)

            # firing rate = spikes / trial / second
            firing_rate = counts / len(rasters) / bin_size
            bin_centers = edges[:-1] + bin_size / 2
            psth_ax.plot(bin_centers, firing_rate, color = cmap(i), lw=2)
            psth_ax.fill_between(bin_centers, firing_rate, alpha=0.3, color=cmap(i))
        
        raster_ax.axvline(0.0, linestyle="--", color="red", linewidth=0.75, alpha=0.5)
        psth_ax.axvline(0.0, linestyle="--", color="red", linewidth=0.75, alpha=0.5)

        raster_ax.set_ylabel("Movement Index")
        psth_ax.set_ylabel("Hz")
        psth_ax.set_xlabel("Time (s)")

        raster_ax.set_title(f"Unit id: {uid}")

    # Turn off unused subplot pairs
    for j in range(n, n_unit_rows * ncols):
        unit_row = j // ncols
        col = j % ncols
        axes[unit_row * 2, col].axis("off")
        axes[unit_row * 2 + 1, col].axis("off")

    plt.suptitle(f"{node} {movement_event} movement: raster and psth")
    plt.tight_layout()
    if save_path:
        plots_dir = Path(save_path) / 'unit_plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / f'{node}_{movement_event}_{n}_units_psth.png'
        plt.savefig(plot_path.as_posix()) # save figure to analyzer path

    plt.show()


def plot_event_modulation(mod_ds, node: str | None = None, epochs: list | None = None,
                          units: list | None = None, vmax: float = 4.0, sort: bool = True,
                          p_source: str = "p_value",
                          star_levels=((0.001, '**'), (0.05, '*')),
                          save_path: Path | str | None = None):
    """Population z-scored event-modulation heatmap for one node (limb).

    Renders units x time as a diverging heatmap, one column per movement epoch
    (e.g. start / max / end), from the dataset produced by
    :func:`neurokinematics.ephys.spikes.modulation.event_modulation`. Each row is
    labelled with its unit id (in the sorted/selected order shown). FDR-significant
    cells are marked with graded stars (``*`` p<0.05, ``**`` p<0.001 by default).

    Args:
        mod_ds (xr.Dataset): Output of ``event_modulation`` (dims
            ``unit, node, epoch, time_bin``).
        node (str | None): Node (limb) to display. Defaults to the first node.
        epochs (list | None): Epoch order to show as columns. Defaults to all
            epochs in ``mod_ds``.
        units (list | None): Subset of unit ids to display. Defaults to all units.
        vmax (float): Symmetric colour limit in z units.
        sort (bool): Order units by mean response z (descending) for the node, so
            modulated units cluster at the top.
        p_source (str): Stars are gated by the FDR-significant flag; this chooses
            which p *grades* them — ``'p_value'`` (default, raw permutation p) or
            ``'p_fdr'``.
        star_levels: ``(threshold, marker)`` pairs for graded stars. Default
            ``((0.001, '**'), (0.05, '*'))``.
        save_path (Path | str | None): If given, save a ``.png`` here (directory
            or full file path).

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    node = node if node is not None else str(mod_ds.node.values[0])
    epochs = list(epochs) if epochs is not None else [str(e) for e in mod_ds.epoch.values]

    sub = mod_ds.sel(node=node)
    if units is not None and len(units):
        sub = sub.sel(unit=list(units))
    unit_ids = np.asarray(sub.unit.values)
    t = np.asarray(sub.time_bin.values)

    if sort:
        # mean response_z across the shown epochs; NaN (untested) units sort last
        rz = sub["response_z"].sel(epoch=epochs).mean("epoch", skipna=True).values
        order = np.argsort(np.where(np.isfinite(rz), rz, -np.inf))[::-1]
    else:
        order = np.arange(len(unit_ids))
    ordered_ids = unit_ids[order]
    n = len(ordered_ids)

    # taller when many units so the per-row unit labels stay legible
    figh = max(3.5, 0.20 * n + 1.0)
    tick_fs = 6 if n > 35 else (7 if n > 18 else 8)
    fig, axes = plt.subplots(1, len(epochs), figsize=(3.4 * len(epochs), figh),
                             sharey=True, squeeze=False)
    axes = axes[0]
    im = None
    for ax, ep in zip(axes, epochs):
        Z = sub["psth_z"].sel(epoch=ep).values[order]
        im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[t[0], t[-1], n - 0.5, -0.5],
                       interpolation="nearest")
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        ax.set_title(str(ep))
        ax.set_xlabel("time from event (s)")
        # graded stars: gated by FDR significance, magnitude from `p_source`
        sig = sub["significant"].sel(epoch=ep).values[order]
        has_p = p_source in mod_ds
        pv = sub[p_source].sel(epoch=ep).values[order] if has_p else None
        for row in np.where(sig)[0]:
            mark = (_p_stars(pv[row], star_levels) or "*") if has_p else "*"
            ax.text(t[-1] * 0.9, row, mark, va="center", ha="center", fontsize=13, color="k")
    # label every row with its unit id (in display order)
    axes[0].set_yticks(np.arange(n))
    axes[0].set_yticklabels([str(int(u)) for u in ordered_ids], fontsize=tick_fs)
    axes[0].set_ylabel("unit id" + (" (sorted)" if sort else ""))
    star_legend = ', '.join(f"{m} p<{thr:g}" for thr, m in sorted(star_levels, reverse=True))
    fig.suptitle(f"Event-aligned z-scored firing — {node}  ({star_legend})")
    if im is not None:
        fig.colorbar(im, ax=axes, label="z (vs baseline)", shrink=0.85)

    if save_path:
        save_path = Path(save_path)
        if save_path.suffix:
            plot_path = save_path
        else:
            save_path.mkdir(parents=True, exist_ok=True)
            plot_path = save_path / f"event_modulation_{node}.png"
        fig.savefig(plot_path.as_posix(), dpi=120, bbox_inches="tight")

    return fig


def _limb_segment(name):
    """'fore' / 'hind' / None from a node name (e.g. 'l_forepaw' -> 'fore')."""
    n = str(name).lower()
    return 'fore' if 'fore' in n else ('hind' if 'hind' in n else None)


def _limb_side(name):
    """'l' / 'r' / None from a node name (e.g. 'r_hindpaw' -> 'r')."""
    n = str(name).lower()
    return 'l' if n.startswith('l') else ('r' if n.startswith('r') else None)


def _limb_styles(nodes):
    """Per-limb plot style: fore/hind -> colour, left/right -> linestyle when names
    parse; otherwise a distinct colour per limb (solid)."""
    seg_color = {'fore': '#14b8a6', 'hind': '#7c3aed'}
    side_ls = {'l': '-', 'r': '--'}
    if all(_limb_segment(n) and _limb_side(n) for n in nodes):
        return {n: dict(color=seg_color[_limb_segment(n)], ls=side_ls[_limb_side(n)]) for n in nodes}
    cyc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return {n: dict(color=cyc[i % len(cyc)], ls='-') for i, n in enumerate(nodes)}


def _p_stars(p, levels=((0.001, '**'), (0.05, '*'))):
    """Significance stars for a p-value: ``'**'`` if p<0.001, ``'*'`` if p<0.05, else ``''``.

    ``levels`` is an iterable of ``(threshold, marker)`` pairs; the marker for the
    smallest threshold the p falls under is returned.
    """
    try:
        p = float(p)
    except (TypeError, ValueError):
        return ''
    if not np.isfinite(p):
        return ''
    for thr, mark in sorted(levels):
        if p < thr:
            return mark
    return ''


def plot_unit_limb_tuning(mod_ds, unit, epochs: list | None = None, nodes: list | None = None,
                          feature: str = "psth_hz", kind: str = "grid",
                          p_source: str = "p_value",
                          star_levels=((0.001, '**'), (0.05, '*')),
                          save_path: Path | str | None = None):
    """Single-unit event-aligned response across limbs and epochs.

    Visualises, for one unit, the PETH produced by
    :func:`neurokinematics.ephys.spikes.modulation.event_modulation` for every
    limb and movement epoch. Significant (limb, epoch) responses are highlighted
    (bold trace + graded stars: ``*`` p<0.05, ``**`` p<0.001 by default). When
    limb names follow the ``{l,r}_{fore,hind}paw`` convention, colour encodes
    fore/hind and linestyle encodes left/right.

    This is the per-unit companion to :func:`plot_event_modulation` — "how does
    *this* neuron respond differently across limbs and movement phases?".

    Args:
        mod_ds (xr.Dataset): Output of ``event_modulation`` (dims
            ``unit, node, epoch, time_bin``).
        unit (int): Unit id to display.
        epochs (list | None): Epoch order (columns). Defaults to all epochs.
        nodes (list | None): Limb order (rows). Defaults to all nodes.
        feature (str): ``'psth_hz'`` (firing rate, default) or ``'psth_z'``
            (z-scored vs baseline).
        p_source (str): Stars are only shown for FDR-significant cells; this
            chooses which p *grades* them — ``'p_value'`` (default, raw permutation
            p) or ``'p_fdr'`` (FDR-corrected; note it cannot reach <0.001 given the
            permutation floor, so ``**`` would not appear).
        star_levels: ``(threshold, marker)`` pairs for graded stars. Default
            ``((0.001, '**'), (0.05, '*'))``.
        kind (str): Layout — ``'grid'`` (default; one small panel per limb×epoch,
            cleanest for a poster), ``'paired'`` (rows = segment fore/hind, columns
            = epoch, overlaying left vs right per panel — best for the laterality
            contrast), ``'overlay'`` (all limbs overlaid per epoch), or
            ``'heatmap'`` (limb×time image, one per epoch; most compact).
        save_path (Path | str | None): If given, save a ``.png`` here (directory
            or full file path).

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    from matplotlib.lines import Line2D

    sub = mod_ds.sel(unit=unit)
    epochs = list(epochs) if epochs is not None else [str(e) for e in mod_ds.epoch.values]
    nodes = list(nodes) if nodes is not None else [str(n) for n in mod_ds.node.values]
    t = np.asarray(mod_ds.time_bin.values)
    styles = _limb_styles(nodes)
    rw = mod_ds.attrs.get('response_window')
    bw = mod_ds.attrs.get('baseline_window')
    ylab = "firing rate (Hz)" if feature == "psth_hz" else "z (vs baseline)"

    def _shade(ax):
        if bw is not None:
            ax.axvspan(bw[0], bw[1], color='0.85', alpha=0.5, lw=0)
        if rw is not None:
            ax.axvspan(rw[0], rw[1], color='#fde68a', alpha=0.5, lw=0)
        ax.axvline(0.0, color='k', lw=0.8, ls=':')

    # stars are gated by the FDR-significant flag (multiple-comparison controlled),
    # then graded by `p_source` (raw permutation p by default — the FDR-corrected p
    # cannot fall below the permutation floor of 1/(n_shuffle+1) once corrected, so
    # ** would never appear if graded on p_fdr).
    has_p = p_source in mod_ds
    def _stars(n, ep):
        if not bool(sub['significant'].sel(node=n, epoch=ep).values):
            return ''
        if not has_p:
            return '*'
        return _p_stars(sub[p_source].sel(node=n, epoch=ep).values, star_levels) or '*'

    if kind == "grid":
        nr, nc = len(nodes), len(epochs)
        fig, axs = plt.subplots(nr, nc, sharex=True, sharey=True,
                                figsize=(2.7 * nc + 0.4, 1.5 * nr + 0.7), squeeze=False)
        for i, n in enumerate(nodes):
            for j, ep in enumerate(epochs):
                ax = axs[i, j]
                _shade(ax)
                y = sub[feature].sel(node=n, epoch=ep).values
                mark = _stars(n, ep)
                sig = bool(mark)
                ax.plot(t, y, color=styles[n]['color'], ls=styles[n]['ls'],
                        lw=2.4 if sig else 1.3, alpha=1.0 if sig else 0.6)
                if mark and rw is not None:
                    ci = int(np.argmin(np.abs(t - 0.5 * (rw[0] + rw[1]))))
                    ax.text(t[ci], y[ci], mark, color=styles[n]['color'], fontsize=13,
                            fontweight='bold', ha='center', va='bottom')
                if i == 0:
                    ax.set_title(str(ep))
                if j == 0:
                    ax.set_ylabel(str(n), fontsize=9, color=styles[n]['color'])
                for sp in ('top', 'right'):
                    ax.spines[sp].set_visible(False)
        for j in range(nc):
            axs[-1, j].set_xlabel("time from event (s)")
        try:
            fig.supylabel(ylab)
        except Exception:
            pass

    elif kind == "paired":
        # group limbs by segment (fore/hind) into rows; overlay left vs right
        segs, seg_members = [], {}
        for n in nodes:
            s = _limb_segment(n) or 'other'
            if s not in segs:
                segs.append(s)
            seg_members.setdefault(s, []).append(n)
        side_color = {'l': '#2563eb', 'r': '#d1495b'}   # left, right
        nr, nc = len(segs), len(epochs)
        fig, axs = plt.subplots(nr, nc, sharex=True, sharey=True,
                                figsize=(3.0 * nc + 0.4, 2.0 * nr + 0.7), squeeze=False)
        for i, seg in enumerate(segs):
            for j, ep in enumerate(epochs):
                ax = axs[i, j]
                _shade(ax)
                for n in seg_members[seg]:
                    col = side_color.get(_limb_side(n), '#444444')
                    mark = _stars(n, ep)
                    sig = bool(mark)
                    y = sub[feature].sel(node=n, epoch=ep).values
                    ax.plot(t, y, color=col, lw=2.4 if sig else 1.3,
                            alpha=1.0 if sig else 0.6)
                    if mark and rw is not None:
                        ci = int(np.argmin(np.abs(t - 0.5 * (rw[0] + rw[1]))))
                        ax.text(t[ci], y[ci], mark, color=col, fontsize=13,
                                fontweight='bold', ha='center', va='bottom')
                if i == 0:
                    ax.set_title(str(ep))
                if j == 0:
                    ax.set_ylabel(f"{seg}paw" if seg in ('fore', 'hind') else str(seg), fontsize=10)
                for sp in ('top', 'right'):
                    ax.spines[sp].set_visible(False)
        for j in range(nc):
            axs[-1, j].set_xlabel("time from event (s)")
        handles = [Line2D([0], [0], color=side_color['l'], lw=2, label='left'),
                   Line2D([0], [0], color=side_color['r'], lw=2, label='right')]
        axs[0, -1].legend(handles=handles, fontsize=8, frameon=False, loc='best')
        try:
            fig.supylabel(ylab)
        except Exception:
            pass

    elif kind == "overlay":
        fig, axes = plt.subplots(1, len(epochs), figsize=(3.6 * len(epochs), 3.8),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ax, ep in zip(axes, epochs):
            _shade(ax)
            for n in nodes:
                y = sub[feature].sel(node=n, epoch=ep).values
                mark = _stars(n, ep)
                sig = bool(mark)
                ax.plot(t, y, color=styles[n]['color'], ls=styles[n]['ls'],
                        lw=2.4 if sig else 1.3, alpha=1.0 if sig else 0.55)
                if mark and rw is not None:
                    ci = int(np.argmin(np.abs(t - 0.5 * (rw[0] + rw[1]))))
                    ax.text(t[ci], y[ci], mark, color=styles[n]['color'], fontsize=13,
                            fontweight='bold', ha='center', va='bottom')
            ax.set_title(str(ep))
            ax.set_xlabel("time from event (s)")
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        axes[0].set_ylabel(ylab)
        handles = [Line2D([0], [0], color=styles[n]['color'], ls=styles[n]['ls'], lw=2, label=str(n))
                   for n in nodes]
        axes[-1].legend(handles=handles, fontsize=8, frameon=False, loc='best')

    elif kind == "heatmap":
        diverging = (feature == "psth_z")
        cmap = "RdBu_r" if diverging else "magma"
        # symmetric limits for z; data-driven for Hz
        if diverging:
            vmax = float(np.nanmax(np.abs(sub[feature].sel(node=nodes, epoch=epochs).values))) or 1.0
            vmin = -vmax
        else:
            vals = sub[feature].sel(node=nodes, epoch=epochs).values
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        fig, axes = plt.subplots(1, len(epochs), figsize=(3.2 * len(epochs), 0.5 * len(nodes) + 1.8),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        im = None
        for ax, ep in zip(axes, epochs):
            Z = np.vstack([sub[feature].sel(node=n, epoch=ep).values for n in nodes])
            im = ax.imshow(Z, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=[t[0], t[-1], len(nodes) - 0.5, -0.5], interpolation="nearest")
            ax.axvline(0.0, color="k", lw=0.8, ls="--")
            ax.set_title(str(ep))
            ax.set_xlabel("time from event (s)")
            for r, n in enumerate(nodes):
                mark = _stars(n, ep)
                if mark:
                    ax.text(t[-1] * 0.9, r, mark, va="center", ha="center", fontsize=13, color="k")
        axes[0].set_yticks(np.arange(len(nodes)))
        axes[0].set_yticklabels([str(n) for n in nodes])
        if im is not None:
            fig.colorbar(im, ax=axes, label=ylab, shrink=0.85)
    else:
        raise ValueError("kind must be 'grid', 'paired', 'overlay', or 'heatmap'")

    star_legend = ', '.join(f"{m} p<{thr:g}" for thr, m in sorted(star_levels, reverse=True))
    fig.suptitle(f"unit {int(unit)} — limb × epoch tuning  ({star_legend})")
    if kind != "heatmap":
        fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        if save_path.suffix:
            plot_path = save_path
        else:
            save_path.mkdir(parents=True, exist_ok=True)
            plot_path = save_path / f"unit_{int(unit)}_limb_tuning.png"
        fig.savefig(plot_path.as_posix(), dpi=120, bbox_inches="tight")

    return fig


# colour + label per laterality pattern, shared by the scatter and the
# across-session stability bar so the categories read identically everywhere.
_PATTERN_ORDER = ['none', 'contra_only', 'ipsi_only',
                  'bilateral_congruent', 'bilateral_opponent']
_PATTERN_STYLE = {
    'none':                ('lightgray', 'n.s.'),
    'contra_only':         ('#999999',   'contra only'),
    'ipsi_only':           ('#555555',   'ipsi only'),
    'bilateral_congruent': ('#3a86ff',   'bilateral · congruent'),
    'bilateral_opponent':  ('#d1495b',   'bilateral · opponent'),
}


def _draw_laterality(ax, axh, df, feature="response_z", lim=None, show_legend=True):
    """Draw the laterality scatter + LI histogram into the given axes.

    Returns a dict of counts (``n``, ``n_bi``, ``n_opp``, ``n_con``) and the
    axis limit used, so callers can build titles / share limits across rows.
    """
    unit_lab = "z (vs baseline)" if feature == "response_z" else "modulation (Hz)"
    x, y = df['contra'].values, df['ipsi'].values
    pattern = df['pattern'].values

    if lim is None:
        lim = np.nanmax(np.abs(np.concatenate([x, y]))) if len(x) else 1.0
        lim = float(lim) * 1.1 or 1.0
    # faint shading of the two opponent quadrants (ipsi/contra opposite signs)
    ax.axhspan(0, lim, xmin=0.0, xmax=0.5, color='#fde68a', alpha=0.18, lw=0)
    ax.axhspan(-lim, 0, xmin=0.5, xmax=1.0, color='#fde68a', alpha=0.18, lw=0)
    ax.axhline(0, color='0.6', lw=0.8)
    ax.axvline(0, color='0.6', lw=0.8)

    SIZE = 40   # all markers the same size — colour alone encodes the category
    for key in _PATTERN_ORDER:
        m = (pattern == key)
        if m.any():
            col, lab = _PATTERN_STYLE[key]
            ax.scatter(x[m], y[m], s=SIZE, c=col, label=lab, edgecolor='none',
                       zorder=3 + key.startswith('bilateral'))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')   # square: x and y on the same scale
    ax.set_xlabel(f"contralateral {unit_lab}")
    ax.set_ylabel(f"ipsilateral {unit_lab}")
    if show_legend:
        ax.legend(fontsize=7, frameon=False, loc='lower right')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    li = df['LI'].dropna().values
    axh.hist(li, bins=np.linspace(-1, 1, 21), color='#6366f1', alpha=0.85)
    axh.axvline(0, color='0.4', lw=0.8, ls='--')
    axh.set_xlabel("laterality index  (+1 ipsi · −1 contra)")
    axh.set_ylabel("units")
    for sp in ('top', 'right'):
        axh.spines[sp].set_visible(False)

    return {
        'lim': lim,
        'n': len(df),
        'n_bi': int(np.nansum(df['bilateral'].values)),
        'n_opp': int(np.sum(pattern == 'bilateral_opponent')),
        'n_con': int(np.sum(pattern == 'bilateral_congruent')),
    }


def _laterality_figure(df, feature="response_z", epoch_lab="all", title="Laterality"):
    """Scatter (contra vs ipsi) + LI histogram from a per-unit laterality table."""
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(9.5, 4.3),
                                  gridspec_kw={'width_ratios': [3, 2]})
    c = _draw_laterality(ax, axh, df, feature=feature, show_legend=True)
    fig.suptitle(f"{title} — epoch: {epoch_lab}   "
                 f"(bilateral {c['n_bi']}/{c['n']}: {c['n_opp']} opponent, "
                 f"{c['n_con']} congruent)")
    fig.tight_layout()
    return fig


def plot_laterality(mod_ds, epoch: str | None = None, ipsi: list | None = None,
                    contra: list | None = None, feature: str = "response_z",
                    units: list | None = None, save_path: Path | str | None = None):
    """Population ipsi-vs-contra (bilateral) tuning summary (scatter + LI histogram).

    Bilateral cells (significant on both sides) are split into opponent (signs
    differ) and congruent (signs match); other cells are ipsi-only / contra-only /
    n.s. All markers are the same size — colour encodes the category.

    Args:
        mod_ds (xr.Dataset | str | Path | pd.DataFrame): An event-modulation
            dataset / zarr path (laterality is computed from it), **or** a
            precomputed per-unit laterality table (e.g. the pooled output of
            :func:`~neurokinematics.ephys.spikes.laterality.laterality_across_sessions`).
        epoch (str | None): Movement epoch to summarise; ``None`` averages epochs.
            Ignored when a precomputed table is passed.
        ipsi (list | None): Ipsilateral limb nodes (default: left-side limbs).
        contra (list | None): Contralateral limb nodes (default: right-side limbs).
        feature (str): ``'response_z'`` (default) or ``'modulation'`` (Hz).
        units (list | None): Subset of unit ids to display. Defaults to all units.
        save_path (Path | str | None): If given, save a ``.png`` here.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    from neurokinematics.ephys.spikes.laterality import laterality

    if isinstance(mod_ds, pd.DataFrame):
        df = mod_ds
        if units is not None and len(units):
            df = df[df['unit'].isin(list(units))]
        title = ("Laterality" if 'session' not in df.columns
                 else f"Laterality ({df['session'].nunique()} sessions)")
    else:
        df = laterality(mod_ds, epoch=epoch, ipsi=ipsi, contra=contra, feature=feature,
                        units=units)
        title = "Laterality"
    epoch_lab = df.attrs.get('epoch', 'all')

    fig = _laterality_figure(df, feature=feature, epoch_lab=epoch_lab, title=title)

    if save_path:
        save_path = Path(save_path)
        if save_path.suffix:
            plot_path = save_path
        else:
            save_path.mkdir(parents=True, exist_ok=True)
            plot_path = save_path / f"laterality_{epoch_lab}.png"
        fig.savefig(plot_path.as_posix(), dpi=120, bbox_inches="tight")

    return fig


def plot_laterality_stability(df, normalize: bool = True, sessions: list | None = None,
                              save_path: Path | str | None = None):
    """Laterality-pattern composition across sessions (stacked bar).

    Shows, per session, the breakdown of units into the laterality patterns
    (opponent / congruent / ipsi-only / contra-only / n.s.) — a reproducibility
    view of the bilateral effect across days. Defaults to **proportions** (so
    differing unit yield across sessions does not distort the comparison); each
    bar is annotated with its unit count.

    Args:
        df (pd.DataFrame): Concatenated per-unit table with ``session`` and
            ``pattern`` columns, e.g. from
            :func:`~neurokinematics.ephys.spikes.laterality.laterality_across_sessions`.
        normalize (bool): Plot fraction of units (default) vs raw counts.
        sessions (list | None): Session order (x-axis). Defaults to sorted unique.
        save_path (Path | str | None): If given, save a ``.png`` here.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    if 'session' not in df.columns:
        raise ValueError("df must have a 'session' column "
                         "(use laterality_across_sessions).")
    sess = list(sessions) if sessions is not None else sorted(df['session'].unique())
    counts = (df.groupby(['session', 'pattern']).size()
                .unstack(fill_value=0)
                .reindex(index=sess, columns=_PATTERN_ORDER, fill_value=0))
    totals = counts.sum(axis=1)
    mat = counts.div(totals.replace(0, np.nan), axis=0).fillna(0) if normalize else counts

    xpos = np.arange(len(sess))
    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * len(sess) + 2.0), 4.4))
    bottom = np.zeros(len(sess))
    for key in _PATTERN_ORDER:
        col, lab = _PATTERN_STYLE[key]
        vals = mat[key].values
        ax.bar(xpos, vals, bottom=bottom, color=col, label=lab, width=0.8,
               edgecolor='white', linewidth=0.4)
        bottom += vals
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(s) for s in sess], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("fraction of units" if normalize else "number of units")
    # annotate N units per session
    for i in range(len(sess)):
        top = (1.0 if normalize else float(totals.iloc[i]))
        ax.text(i, top + (0.012 if normalize else 0.01 * max(totals.max(), 1)),
                f"n={int(totals.iloc[i])}", ha='center', va='bottom',
                fontsize=7, color='0.3')
    if normalize:
        ax.set_ylim(0, 1.14)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.22))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title("Laterality composition across sessions")
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plot_path = save_path if save_path.suffix else (save_path / "laterality_stability.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path.as_posix(), dpi=120, bbox_inches="tight")

    return fig


def plot_laterality_epochs(source, epochs=("start", "max", "end"), feature="response_z",
                           ipsi: list | None = None, contra: list | None = None,
                           save_path: Path | str | None = None):
    """Laterality (scatter + LI histogram) faceted by movement epoch — one row each.

    Avoids averaging across epochs: each row is the ipsi-vs-contra summary for a
    single epoch, with a shared axis limit so rows are directly comparable.

    Args:
        source: Either an event-modulation dataset / zarr path (laterality is
            computed per epoch via
            :func:`~neurokinematics.ephys.spikes.laterality.laterality`), **or** a
            ``{epoch: laterality_table}`` mapping of precomputed per-epoch tables
            — e.g. one pooled
            :func:`~neurokinematics.ephys.spikes.laterality.laterality_across_sessions`
            table per epoch (the multi-session form).
        epochs (tuple): Epochs to show as rows, top to bottom.
        feature (str): ``'response_z'`` (default) or ``'modulation'`` (Hz).
        ipsi / contra (list | None): Limb node lists (only used when computing).
        save_path (Path | str | None): If given, save a ``.png`` here.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    from neurokinematics.ephys.spikes.laterality import laterality

    if isinstance(source, pd.DataFrame):
        raise ValueError(
            "A single DataFrame is one epoch's table — pass a {epoch: table} mapping "
            "(one per epoch) or an event-modulation dataset to facet by epoch.")
    if isinstance(source, dict):
        dfs = {ep: source[ep] for ep in epochs}
    else:
        dfs = {ep: laterality(source, epoch=ep, ipsi=ipsi, contra=contra, feature=feature)
               for ep in epochs}

    # shared axis limit so the rows are comparable
    allv = [np.abs(np.concatenate([dfs[ep]['contra'].values, dfs[ep]['ipsi'].values]))
            for ep in epochs if len(dfs[ep])]
    lim = float(np.nanmax(np.concatenate(allv))) * 1.1 if allv else 1.0
    lim = lim or 1.0

    n_ep = len(epochs)
    fig, axs = plt.subplots(n_ep, 2, figsize=(9.0, 3.3 * n_ep),
                            gridspec_kw={'width_ratios': [3, 2]}, squeeze=False)
    for i, ep in enumerate(epochs):
        ax, axh = axs[i, 0], axs[i, 1]
        c = _draw_laterality(ax, axh, dfs[ep], feature=feature, lim=lim,
                             show_legend=(i == 0))
        ax.set_title(f"{ep}   (bilateral {c['n_bi']}/{c['n']}: "
                     f"{c['n_opp']} opp, {c['n_con']} con)", fontsize=10)
        if i < n_ep - 1:            # only the bottom row keeps x-labels
            ax.set_xlabel("")
            axh.set_xlabel("")
    fig.suptitle("Laterality across movement epochs")
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plot_path = save_path if save_path.suffix else (save_path / "laterality_epochs.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path.as_posix(), dpi=120, bbox_inches="tight")

    return fig


def _sig_stars3(p):
    """Three-level significance string: *** <0.001, ** <0.01, * <0.05, else 'n.s.'."""
    if p is None or not np.isfinite(p):
        return 'n.s.'
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


def plot_laterality_summary(tables, epochs=("start", "max", "end"),
                            save_path: Path | str | None = None):
    """Single summary plot of the two laterality contrasts across epochs.

    For each epoch, bars give the mean fraction of units (of all units) that are
    **ipsi-only**, **contra-only**, **opponent**, and **congruent** across
    sessions (± SD, with per-session points overlaid). Two paired comparisons are
    annotated per epoch — **ipsi-only vs contra-only** and **opponent vs
    congruent** (Wilcoxon signed-rank across sessions; * p<0.05, ** p<0.01,
    *** p<0.001) — so a single figure shows both the contra/ipsi balance and the
    opponent/congruent balance, and how each changes across movement phase.

    Args:
        tables: ``{epoch: laterality_table}`` mapping (pooled per-unit tables with
            a ``session`` column), or a single table (one epoch).
        epochs (tuple): Epoch order along the x-axis.
        save_path (Path | str | None): If given, save a ``.png`` here.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    from scipy import stats as _ss
    from neurokinematics.ephys.spikes.laterality import laterality_stats

    if isinstance(tables, pd.DataFrame):
        tables = {str(tables.attrs.get('epoch', 'all')): tables}
    epochs = [e for e in epochs if e in tables] or list(tables.keys())
    ps = laterality_stats(tables, epochs=epochs)['per_session']

    cats = ['ipsi_only', 'contra_only', 'bilateral_opponent', 'bilateral_congruent']
    fcol = {'ipsi_only': 'ipsi_only_frac', 'contra_only': 'contra_only_frac',
            'bilateral_opponent': 'opponent_frac', 'bilateral_congruent': 'congruent_frac'}
    offs = np.array([-0.30, -0.10, 0.10, 0.30])
    width = 0.18

    def _wil(a, b):
        d = np.asarray(a, float) - np.asarray(b, float)
        d = d[np.isfinite(d)]
        d = d[d != 0]
        if d.size < 1:
            return float('nan')
        try:
            return float(_ss.wilcoxon(d).pvalue)
        except Exception:
            return float('nan')

    fig, ax = plt.subplots(figsize=(max(6.0, 2.4 * len(epochs) + 2.0), 4.8))
    rng = np.random.default_rng(0)
    ymax = 0.0
    pair_p = []
    for i, ep in enumerate(epochs):
        sub = ps[ps['epoch'] == ep]
        for j, cat in enumerate(cats):
            vals = sub[fcol[cat]].values
            vals = vals[np.isfinite(vals)]
            m = float(np.mean(vals)) if vals.size else 0.0
            sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            col, lab = _PATTERN_STYLE[cat]
            x = i + offs[j]
            ax.bar(x, m, width=width, color=col, edgecolor='white', linewidth=0.4,
                   label=(lab if i == 0 else None), zorder=2)
            ax.errorbar(x, m, yerr=sd, fmt='none', ecolor='0.3', capsize=2, lw=0.8, zorder=3)
            if vals.size:
                xs = x + (rng.random(vals.size) - 0.5) * width * 0.6
                ax.scatter(xs, vals, s=8, color='0.2', alpha=0.5, zorder=4)
                ymax = max(ymax, m + sd, float(vals.max()))
        pair_p.append((i,
                       _wil(sub['ipsi_only_frac'].values, sub['contra_only_frac'].values),
                       _wil(sub['opponent_frac'].values, sub['congruent_frac'].values)))

    # significance brackets: ipsi-vs-contra (bars 0,1) and opponent-vs-congruent (bars 2,3)
    yb = ymax * 1.06 if ymax else 1.0
    h = (ymax or 1.0) * 0.02
    for i, p_ic, p_oc in pair_p:
        for (j_left, j_right), p in (((0, 1), p_ic), ((2, 3), p_oc)):
            x1, x2 = i + offs[j_left], i + offs[j_right]
            ax.plot([x1, x1, x2, x2], [yb, yb + h, yb + h, yb], color='k', lw=0.9)
            ax.text((x1 + x2) / 2, yb + h, _sig_stars3(p), ha='center', va='bottom', fontsize=10)

    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels([str(e) for e in epochs])
    ax.set_ylabel("fraction of units")
    ax.set_ylim(0, (ymax or 1.0) * 1.22)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc='upper left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.set_title("Unilateral (ipsi vs contra) and bilateral (opponent vs congruent) "
                 "tuning across epochs", fontsize=10)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plot_path = save_path if save_path.suffix else (save_path / "laterality_summary.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path.as_posix(), dpi=120, bbox_inches="tight")

    return fig


def plot_decode_summary(summary, save_path: Path | str | None = None):
    """Across-session population-decode summary (mean CV R2 per decode type).

    Bars give the mean cross-validated R2 across sessions (+/- SD, with per-session
    points overlaid) for each decode in the table — typically ``contra``, ``ipsi``,
    and ``ipsi_residual`` (ipsi orthogonal to contra, the co-movement control).
    When a shuffle null was run, each bar is annotated with "k/N" sessions reaching
    p < 0.05.

    Args:
        summary (pd.DataFrame | list): Output of
            ``neurokinematics.models.glm.decode_across_sessions``, or a list of such
            DataFrames (concatenated automatically).
        save_path (Path | str | None): If given, save a ``.png`` here.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    if isinstance(summary, (list, tuple)):
        summary = pd.concat(list(summary), ignore_index=True)

    present = list(summary['decode'].unique())
    order = [d for d in ('contra', 'ipsi', 'ipsi_residual') if d in present]
    order += [d for d in present if d not in order]
    colours = {'contra': '#d1495b', 'ipsi': '#2563eb', 'ipsi_residual': '#7c3aed'}
    label_map = {'ipsi_residual': 'ipsi ⊥ contra'}

    fig, ax = plt.subplots(figsize=(max(4.5, 1.7 * len(order) + 1.5), 4.4))
    rng = np.random.default_rng(0)
    ymax, ymin = 0.0, 0.0
    for i, d in enumerate(order):
        sub = summary[summary['decode'] == d]
        vals = np.asarray(sub['cv_r2'].values, dtype=float)
        vals = vals[np.isfinite(vals)]
        m = float(np.mean(vals)) if vals.size else 0.0
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        ax.bar(i, m, width=0.6, color=colours.get(d, '#888888'),
               edgecolor='white', linewidth=0.4, zorder=2)
        ax.errorbar(i, m, yerr=sd, fmt='none', ecolor='0.3', capsize=3, lw=0.9, zorder=3)
        if vals.size:
            xs = i + (rng.random(vals.size) - 0.5) * 0.3
            ax.scatter(xs, vals, s=12, color='0.2', alpha=0.6, zorder=4)
            ymax = max(ymax, m + sd, float(vals.max()))
            ymin = min(ymin, float(vals.min()))
        if 'shuffle_p' in sub.columns:
            p = np.asarray(sub['shuffle_p'].values, dtype=float)
            p = p[np.isfinite(p)]
            if p.size:
                ax.text(i, m + sd + 0.02 * (ymax or 1),
                        f"{int((p < 0.05).sum())}/{p.size}",
                        ha='center', va='bottom', fontsize=9, color='0.3')
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label_map.get(d, d) for d in order])
    ax.set_ylabel("CV R2  (mean +/- SD across sessions)")
    ax.set_ylim(min(0, ymin * 1.1), (ymax or 1.0) * 1.2)
    ax.set_title("Population decode across sessions")
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plot_path = save_path if save_path.suffix else (save_path / "decode_summary.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path.as_posix(), dpi=120, bbox_inches="tight")

    return fig


def plot_decode_example(outputs, kind: str = "concat", n_events: int = 6,
                        color: str = "#6366f1", title: str | None = None,
                        save_path: Path | str | None = None):
    """Decoded vs actual kinematic from a fitted decoder — an example trace.

    Reconstructs the (event × time) grids from ``create_glm_decoder`` outputs and
    overlays the actual kinematic with the (cross-validated, if CV was on) decoded
    one. ``kind='concat'`` strings a few example movements end-to-end (the most
    intuitive "the population tracks the limb" view); ``kind='average'`` shows the
    trial-averaged decode over the movement window (± SEM).

    Args:
        outputs (dict): Third return value of
            :func:`neurokinematics.models.glm.create_glm_decoder`.
        kind (str): ``'concat'`` (default) or ``'average'``.
        n_events (int): Number of example movements to concatenate (``'concat'``).
        color (str): Colour for the decoded trace.
        title (str | None): Axis title / y-label stem (e.g. "ipsi hindpaw speed").
        save_path (Path | str | None): If given, save a ``.png`` here.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    ev = np.asarray(outputs['event_idx'], dtype=int)
    ti = np.asarray(outputs['time_idx'], dtype=int)
    obs = np.asarray(outputs['observed'], dtype=float)
    pred = np.asarray(outputs['predicted'], dtype=float)
    tb = np.asarray(outputs['time_bins'], dtype=float)
    n_time = len(tb)
    n_ev = int(ev.max()) + 1 if ev.size else 0
    O = np.full((n_ev, n_time), np.nan)
    P = np.full((n_ev, n_time), np.nan)
    O[ev, ti] = obs
    P[ev, ti] = pred
    r2 = (outputs.get('params', {}) or {}).get('metrics', {}).get('cv_r2')

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    if kind == "average":
        n = np.sum(np.isfinite(O), axis=0)
        om, pm = np.nanmean(O, axis=0), np.nanmean(P, axis=0)
        osem = np.nanstd(O, axis=0) / np.sqrt(np.maximum(n, 1))
        ax.fill_between(tb, om - osem, om + osem, color='0.7', alpha=0.4, lw=0)
        ax.plot(tb, om, color='0.2', lw=2, label='actual')
        ax.plot(tb, pm, color=color, lw=2, label='decoded')
        ax.set_xlabel("time from movement onset (s)")
    else:
        picks = np.argsort(np.sum(np.isfinite(O), axis=1))[::-1][:n_events]
        seg_o, seg_p, seg_x, bounds, x = [], [], [], [], 0
        for e in picks:
            m = np.isfinite(O[e]) & np.isfinite(P[e])
            k = int(m.sum())
            if not k:
                continue
            seg_o.append(O[e][m]); seg_p.append(P[e][m])
            seg_x.append(np.arange(x, x + k)); x += k; bounds.append(x)
        if seg_o:
            xs = np.concatenate(seg_x)
            ax.plot(xs, np.concatenate(seg_o), color='0.2', lw=1.6, label='actual')
            ax.plot(xs, np.concatenate(seg_p), color=color, lw=1.6, label='decoded')
            for b in bounds[:-1]:
                ax.axvline(b - 0.5, color='0.85', lw=0.6)
        ax.set_xlabel("time (bins; example movements concatenated)")

    ax.set_ylabel(title or "kinematic")
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ttl = title or "decode"
    if r2 is not None and np.isfinite(r2):
        ttl += f"  (CV R² = {r2:.2f})"
    ax.set_title(ttl)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        plot_path = save_path if save_path.suffix else (save_path / "decode_example.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path.as_posix(), dpi=120, bbox_inches="tight")

    return fig