"""Ipsilateral-vs-contralateral (laterality) tuning from event modulation.

Reduces the per-(unit, limb, epoch) table produced by
:func:`neurokinematics.ephys.spikes.modulation.event_modulation` to a per-unit
ipsi-vs-contra summary: each unit's mean modulation for ipsilateral limbs, for
contralateral limbs, a laterality index, and a flag for the "bilateral" pattern
(significant increase for ipsilateral movements *and* significant decrease for
contralateral ones) — the integration-across-the-body-axis result.

Ipsi/contra are assigned by limb side relative to the recorded hemisphere. With a
left-hemisphere probe the **left** limbs are ipsilateral; this is the default
inference (node names beginning with ``l`` = ipsi, ``r`` = contra). Pass explicit
``ipsi`` / ``contra`` node lists to override.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from neurokinematics.io import load_zarr, save_dataframe


def _infer_sides(nodes):
    """Split node names into (left, right) by leading letter."""
    left = [n for n in nodes if str(n).lower().startswith('l')]
    right = [n for n in nodes if str(n).lower().startswith('r')]
    return left, right


def laterality(mod_ds, epoch: str | None = None, ipsi: list | None = None,
               contra: list | None = None, feature: str = "response_z",
               units: list | None = None,
               save_path: Path | str | None = None) -> pd.DataFrame:
    """Per-unit ipsi-vs-contra tuning summary from an event-modulation dataset.

    Args:
        mod_ds (xr.Dataset | str | Path): Output of
            :func:`~neurokinematics.ephys.spikes.modulation.event_modulation`, or a
            path to its ``event_modulation.zarr``.
        epoch (str | None): Movement epoch to summarise (e.g. ``'start'``). ``None``
            (default) averages each side's effect across all epochs.
        ipsi (list | None): Ipsilateral limb node names. Defaults to left-side limbs.
        contra (list | None): Contralateral limb node names. Defaults to right-side.
        feature (str): Per-cell effect to aggregate — ``'response_z'`` (default,
            baseline-normalised, comparable across units) or ``'modulation'``
            (signed Hz).
        units (list | None): Subset of unit ids to keep. Defaults to all units.
        save_path (Path | str | None): If given, write ``laterality.csv`` here
            (directory) or to the given file path.

    Returns:
        pd.DataFrame: One row per unit with ``unit``, ``ipsi`` / ``contra`` (mean
        signed effect), ``LI`` (laterality index in [-1, 1], >0 = ipsi-preferring),
        ``ipsi_sig`` / ``contra_sig`` (any contributing cell significant),
        ``bilateral`` (significant on *both* sides, any sign), ``pattern``
        (``'bilateral_opponent'`` | ``'bilateral_congruent'`` | ``'ipsi_only'`` |
        ``'contra_only'`` | ``'none'``), and ``ipsi_up_contra_down`` (the specific
        ipsi↑/contra↓ opponent subtype). ``.attrs`` carries the ``epoch``,
        ``feature``, and ipsi/contra node lists used.

    Example:
        >>> df = laterality(mod_ds, epoch='start')
        >>> int(df['bilateral'].sum())                       # modulated by both sides
        >>> df['pattern'].value_counts()                     # opponent vs congruent vs ...
    """
    if isinstance(mod_ds, (str, Path)):
        mod_ds = load_zarr(mod_ds, method='xarray')

    nodes = [str(n) for n in mod_ds.node.values]
    if ipsi is None or contra is None:
        left, right = _infer_sides(nodes)
        ipsi = list(ipsi) if ipsi is not None else left
        contra = list(contra) if contra is not None else right
    if not ipsi or not contra:
        raise ValueError(
            "Could not determine ipsi/contra limbs from node names "
            f"{nodes}. Pass explicit ipsi=[...] and contra=[...].")

    if epoch is not None:
        sel = mod_ds.sel(epoch=epoch)
    else:
        sel = mod_ds
    mod = sel[feature]
    sig = sel['significant']

    def _side_stats(side_nodes):
        m = mod.sel(node=side_nodes)
        s = sig.sel(node=side_nodes)
        reduce_dims = [d for d in m.dims if d != 'unit']
        mean_eff = m.mean(dim=reduce_dims, skipna=True).values
        any_sig = (s.sum(dim=[d for d in s.dims if d != 'unit']) > 0).values
        return mean_eff, any_sig

    ipsi_eff, ipsi_sig = _side_stats(ipsi)
    contra_eff, contra_sig = _side_stats(contra)

    denom = np.abs(ipsi_eff) + np.abs(contra_eff)
    with np.errstate(invalid='ignore', divide='ignore'):
        li = np.where(denom > 0, (ipsi_eff - contra_eff) / denom, np.nan)

    # bilateral = significantly modulated by BOTH sides (any sign combination).
    # The sign combination is captured separately by `pattern`:
    #   bilateral_opponent  – ipsi and contra change in opposite directions
    #   bilateral_congruent – ipsi and contra change in the same direction
    both = ipsi_sig & contra_sig
    same_sign = np.sign(ipsi_eff) == np.sign(contra_eff)
    pattern = np.full(ipsi_eff.shape, 'none', dtype=object)
    pattern[ipsi_sig & ~contra_sig] = 'ipsi_only'
    pattern[contra_sig & ~ipsi_sig] = 'contra_only'
    pattern[both & same_sign] = 'bilateral_congruent'
    pattern[both & ~same_sign] = 'bilateral_opponent'
    # the specific pattern named in the abstract (a subtype of opponent)
    ipsi_up_contra_down = both & (ipsi_eff > 0) & (contra_eff < 0)

    df = pd.DataFrame({
        'unit': np.asarray(mod_ds.unit.values),
        'ipsi': ipsi_eff,
        'contra': contra_eff,
        'LI': li,
        'ipsi_sig': ipsi_sig,
        'contra_sig': contra_sig,
        'bilateral': both,
        'pattern': pattern,
        'ipsi_up_contra_down': ipsi_up_contra_down,
    })
    if units is not None and len(units):
        df = df[df['unit'].isin(list(units))].reset_index(drop=True)

    df.attrs.update({
        'epoch': 'all' if epoch is None else str(epoch),
        'feature': feature,
        'ipsi_nodes': list(ipsi),
        'contra_nodes': list(contra),
    })

    if save_path:
        save_path = Path(save_path)
        out = save_path if save_path.suffix else (save_path / 'laterality.csv')
        out.parent.mkdir(parents=True, exist_ok=True)
        save_dataframe(df, out, storage_format='csv')

    return df


def _session_phy_folder(spikes_dir):
    """Locate ``<spikes>/*/phy_output`` containing cluster_group.tsv, or None."""
    if not spikes_dir:
        return None
    hits = list(Path(spikes_dir).glob('*/phy_output/cluster_group.tsv'))
    return hits[0].parent if hits else None


def _resolve_mod_file(label, spikes_dir, mod_file):
    """Pick one session's event-modulation store under ``<spikes>/modulation/``.

    With ``mod_file=None`` the *newest* ``event_modulation*.zarr`` is used (so
    timestamped runs from :func:`event_modulation` resolve automatically). A
    ``str`` (filename or full path), a ``{label: filename_or_path}`` mapping, or a
    ``callable(label) -> filename_or_path`` pins a specific store per session; a
    bare filename is looked up inside the session's modulation folder. Returns the
    resolved ``Path`` or ``None`` if nothing matches.
    """
    if not spikes_dir:
        return None
    mod_dir = Path(spikes_dir) / 'modulation'
    spec = None
    if mod_file is not None:
        spec = (mod_file(label) if callable(mod_file)
                else mod_file.get(label) if isinstance(mod_file, dict)
                else mod_file)
    if spec:
        p = Path(spec)
        if not p.exists():                         # treat as a bare filename
            p = mod_dir / Path(spec).name
        return p if p.exists() else None
    if not mod_dir.exists():
        return None
    files = sorted(mod_dir.glob('event_modulation*.zarr'),
                   key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _iter_session_mods(sessions, mod_file=None):
    """Normalise the ``sessions`` argument to ``(label, mod, spikes_dir)`` triples.

    Accepts a ``{label: mod_ds_or_zarr}`` mapping, an iterable of
    ``(label, mod_ds_or_zarr)`` pairs, an ``ExperimentSubject``-like object with a
    ``.sessions`` attribute, or a single session-like object with ``.dirs``. For
    the latter two, each session's event-modulation store is located under
    ``<spikes>/modulation/`` via :func:`_resolve_mod_file` (``mod_file`` selects
    *which* store — newest by default) and ``spikes_dir`` is carried through so
    good-unit curation labels can be resolved; for mapping / pair inputs the value
    is used directly and ``spikes_dir`` is ``None`` (so ``mod_file`` is ignored —
    those inputs are already explicit).
    """
    if isinstance(sessions, dict):
        return [(lbl, mod, None) for lbl, mod in sessions.items()]

    if hasattr(sessions, 'sessions'):              # subject-like
        out = []
        for sess in (sessions.sessions or []):
            label = str(getattr(sess, 'session_id', sess))
            sd = getattr(sess, 'dirs', {}).get('spikes')
            p = _resolve_mod_file(label, sd, mod_file)
            if p is not None:
                out.append((label, p, sd))
        return out

    if hasattr(sessions, 'dirs'):                  # single session-like
        label = str(getattr(sessions, 'session_id', 'session'))
        sd = getattr(sessions, 'dirs', {}).get('spikes')
        p = _resolve_mod_file(label, sd, mod_file)
        return [(label, p, sd)] if p is not None else []

    # iterable of (label, mod) pairs
    return [(lbl, mod, None) for lbl, mod in sessions]


def _resolve_good_units(label, spikes_dir, good_units):
    """Resolve the unit subset for one session, or None for 'all units'."""
    if good_units is None:
        return None
    if good_units == "auto":
        phy = _session_phy_folder(spikes_dir)
        if phy is None:
            raise ValueError(
                f"good_units='auto' but no phy cluster_group.tsv found for session "
                f"'{label}'. Curate spikes, pass session objects, or supply an "
                f"explicit {{session: [unit_ids]}} mapping.")
        from neurokinematics.ephys.spikes.curation import good_unit_ids
        return good_unit_ids(phy)
    if callable(good_units):
        return list(good_units(label))
    if isinstance(good_units, dict):
        return good_units.get(label)
    return list(good_units)                        # same list for every session


def laterality_across_sessions(sessions, select=None, good_units=None,
                               epoch: str | None = None,
                               ipsi: list | None = None, contra: list | None = None,
                               feature: str = "response_z",
                               mod_file=None,
                               save_path: Path | str | None = None) -> pd.DataFrame:
    """Per-unit laterality pooled across sessions (one animal).

    Runs :func:`laterality` on each session's event-modulation dataset and
    concatenates the per-unit tables, adding a ``session`` column. Treats sessions
    as independent populations (units are *not* assumed tracked across days), so
    rows are unit-sessions — appropriate for a within-animal, multi-session summary.

    Args:
        sessions: A ``{label: mod_ds_or_zarr}`` mapping (pass explicit zarr paths
            here for full per-session control), an iterable of
            ``(label, mod_ds_or_zarr)`` pairs, or an ``ExperimentSubject`` /
            session object (its sessions' ``event_modulation*.zarr`` stores are
            located automatically — see ``mod_file``).
        select: Optional subset of sessions to include — a list/set of session
            labels to keep, or a ``callable(label) -> bool``. ``None`` (default)
            uses every session found.
        good_units: Restrict to curation-approved units. ``None`` (default) keeps
            all units; ``'auto'`` reads each session's phy ``cluster_group.tsv``
            and keeps the ``good`` clusters (requires session objects / a Subject);
            a ``{session_label: [unit_ids]}`` mapping or ``callable(label) ->
            unit_ids`` supplies them explicitly.
        epoch (str | None): Movement epoch to summarise; ``None`` averages epochs.
        ipsi / contra (list | None): Limb node lists (default: left = ipsi).
        feature (str): ``'response_z'`` (default) or ``'modulation'``.
        mod_file: When ``sessions`` is a Subject/session object, selects *which*
            timestamped ``event_modulation*.zarr`` to use per session. ``None``
            (default) takes the newest store in each session; a ``str`` (filename
            or path) applies to every session; a ``{label: filename_or_path}``
            mapping or ``callable(label) -> filename_or_path`` pins one per session.
            Unlike a ``{label: zarr}`` ``sessions`` mapping, this keeps the session
            objects so ``good_units='auto'`` still resolves. Ignored for mapping /
            pair inputs (already explicit).
        save_path (Path | str | None): If given, write ``laterality_sessions.csv``.

    Returns:
        pd.DataFrame: Concatenated per-unit table with a leading ``session``
        column; ``.attrs`` carries ``epoch``, ``feature``, and ``n_sessions``.

    Example:
        >>> pooled = laterality_across_sessions(subject, epoch='start')
        >>> pooled.groupby('session')['bilateral'].mean()   # bilateral fraction / session
    """
    items = _iter_session_mods(sessions, mod_file=mod_file)

    if select is not None:
        if callable(select):
            items = [(lbl, m, sd) for lbl, m, sd in items if select(lbl)]
        else:
            keep = {str(s) for s in select}
            items = [(lbl, m, sd) for lbl, m, sd in items if str(lbl) in keep]

    if not items:
        raise ValueError(
            "No sessions to summarise. Either none had an event_modulation.zarr, "
            "or `select` filtered them all out.")

    frames = []
    for label, mod, spikes_dir in items:
        units = _resolve_good_units(label, spikes_dir, good_units)
        d = laterality(mod, epoch=epoch, ipsi=ipsi, contra=contra,
                       feature=feature, units=units)
        d.insert(0, 'session', str(label))
        frames.append(d)

    out = pd.concat(frames, ignore_index=True)
    out.attrs.update({
        'epoch': 'all' if epoch is None else str(epoch),
        'feature': feature,
        'n_sessions': len(frames),
    })

    if save_path:
        save_path = Path(save_path)
        target = save_path if save_path.suffix else (save_path / 'laterality_sessions.csv')
        target.parent.mkdir(parents=True, exist_ok=True)
        save_dataframe(out, target, storage_format='csv')

    return out


def _wilcoxon_vs(values, mu):
    """One-sample Wilcoxon signed-rank of finite ``values`` against ``mu``.

    Returns ``(p, n_sessions, median)``; ``p`` is NaN when there are too few
    non-tied observations to test.
    """
    from scipy import stats
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    med = float(np.median(v)) if n else float('nan')
    d = v - mu
    d = d[d != 0]
    if d.size < 1:
        return float('nan'), n, med
    try:
        p = float(stats.wilcoxon(d).pvalue)
    except Exception:
        p = float('nan')
    return p, n, med


def _spearman_trend(x, y):
    """Spearman correlation of ``y`` vs ``x`` over finite pairs.

    Returns ``(rho, p, n)``; NaN when fewer than 3 usable points.
    """
    from scipy import stats
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return float('nan'), float('nan'), n
    try:
        res = stats.spearmanr(x[m], y[m])
        return float(res.correlation), float(res.pvalue), n
    except Exception:
        return float('nan'), float('nan'), n


def laterality_stats(tables, epochs: list | None = None,
                     session_order: list | None = None) -> dict:
    """Session-level summary and non-parametric tests of laterality composition.

    Treats **sessions as the unit of replication** (appropriate for a single
    animal): computes per-session proportions for each epoch, then tests those
    across sessions. Also reports pooled binomial p-values as a descriptive
    reference — these pool unit-sessions and are *not* independent, so cite the
    session-level Wilcoxon/Friedman for inference.

    Args:
        tables: A ``{epoch: laterality_table}`` mapping (each table the pooled
            per-unit output of :func:`laterality_across_sessions`, with a
            ``session`` column), or a single per-unit table (treated as one epoch).
        epochs (list | None): Epoch order to analyse. Defaults to the mapping keys.
        session_order (list | None): Session labels in chronological order, used
            for the across-session trend test. Defaults to sorted label order
            (fine when labels are dates).

    Returns:
        dict with:
          * ``per_session`` (DataFrame): one row per (session, epoch) with counts
            and proportions — of all units (``bilateral_frac``, ``opponent_frac``,
            ``congruent_frac``, ``ipsi_only_frac``, ``contra_only_frac``) and of
            subsets (``opp_frac_bilateral``, ``contra_frac_unilateral``), plus
            ``median_LI``.
          * ``within_epoch`` (DataFrame): per epoch, Wilcoxon p-values for
            "opponent > congruent" (opp_frac vs 0.5), "contra > ipsi"
            (contra_frac vs 0.5) and "LI ≠ 0", plus the pooled binomial references.
          * ``across_epoch`` (DataFrame): per metric, a Friedman test across the
            epochs (start/max/end) — does that proportion differ between epochs
            (requires ≥3 epochs and ≥3 complete sessions).
          * ``across_epoch_posthoc`` (DataFrame): pairwise Wilcoxon signed-rank
            between epoch pairs per metric, Holm-corrected (``wilcoxon_p_holm``) —
            *which* epochs differ.
          * ``session_trend`` (DataFrame): per (epoch, metric), the **Spearman
            correlation of the per-session proportion vs session order** — i.e.
            does the representation (e.g. ``bilateral_frac``, ``contra_only_frac``)
            change across days. Note these are *independent* unit populations per
            day, so a trend reflects population-level change (and may also reflect
            behavioural/drift/yield changes), not the same cells over time.

    Example:
        >>> tables = {ep: laterality_across_sessions(subject, epoch=ep) for ep in
        ...           ("start", "max", "end")}
        >>> stats = laterality_stats(tables)
        >>> stats["within_epoch"]
        >>> stats["per_session"].to_csv("laterality_per_session.csv", index=False)
    """
    from scipy import stats

    if isinstance(tables, pd.DataFrame):
        tables = {str(tables.attrs.get('epoch', 'all')): tables}
    if epochs is None:
        epochs = list(tables.keys())

    rows = []
    for ep in epochs:
        df = tables[ep]
        if 'session' not in df.columns:
            df = df.assign(session='all')
        for sess, g in df.groupby('session'):
            pat = g['pattern'].values
            n = len(g)
            n_opp = int((pat == 'bilateral_opponent').sum())
            n_con = int((pat == 'bilateral_congruent').sum())
            n_ipsi = int((pat == 'ipsi_only').sum())
            n_contra = int((pat == 'contra_only').sum())
            n_bi, n_uni = n_opp + n_con, n_ipsi + n_contra
            rows.append({
                'session': sess, 'epoch': ep, 'n_units': n,
                'n_opponent': n_opp, 'n_congruent': n_con,
                'n_ipsi_only': n_ipsi, 'n_contra_only': n_contra,
                'n_bilateral': n_bi, 'n_ns': int((pat == 'none').sum()),
                # fractions of ALL units this session (for across-session trends)
                'bilateral_frac': (n_bi / n) if n else np.nan,
                'opponent_frac': (n_opp / n) if n else np.nan,
                'congruent_frac': (n_con / n) if n else np.nan,
                'ipsi_only_frac': (n_ipsi / n) if n else np.nan,
                'contra_only_frac': (n_contra / n) if n else np.nan,
                # composition fractions (within bilateral / within unilateral)
                'opp_frac_bilateral': (n_opp / n_bi) if n_bi else np.nan,
                'contra_frac_unilateral': (n_contra / n_uni) if n_uni else np.nan,
                'median_LI': float(np.nanmedian(g['LI'].values)) if n else np.nan,
            })
    per_session = pd.DataFrame(rows)

    within = []
    for ep in epochs:
        sub = per_session[per_session['epoch'] == ep]
        p_opp, n_s, med_opp = _wilcoxon_vs(sub['opp_frac_bilateral'], 0.5)
        p_con, _, med_con = _wilcoxon_vs(sub['contra_frac_unilateral'], 0.5)
        p_li, _, med_li = _wilcoxon_vs(sub['median_LI'], 0.0)

        pat = tables[ep]['pattern'].values          # pooled (descriptive only)
        n_opp = int((pat == 'bilateral_opponent').sum())
        n_con = int((pat == 'bilateral_congruent').sum())
        n_contra = int((pat == 'contra_only').sum())
        n_ipsi = int((pat == 'ipsi_only').sum())
        binom_opp = (float(stats.binomtest(n_opp, n_opp + n_con, 0.5).pvalue)
                     if (n_opp + n_con) else float('nan'))
        binom_con = (float(stats.binomtest(n_contra, n_contra + n_ipsi, 0.5).pvalue)
                     if (n_contra + n_ipsi) else float('nan'))
        within.append({
            'epoch': ep, 'n_sessions': len(sub),
            'opp_frac_median': med_opp, 'opp_vs_congruent_wilcoxon_p': p_opp,
            'contra_frac_median': med_con, 'contra_vs_ipsi_wilcoxon_p': p_con,
            'median_LI': med_li, 'LI_vs_zero_wilcoxon_p': p_li,
            'pooled_n_opponent': n_opp, 'pooled_n_congruent': n_con,
            'pooled_opp_vs_con_binom_p': binom_opp,
            'pooled_n_contra_only': n_contra, 'pooled_n_ipsi_only': n_ipsi,
            'pooled_contra_vs_ipsi_binom_p': binom_con,
        })
    within_epoch = pd.DataFrame(within)

    # ── differences across epochs (within-session, paired) ───────────────────
    # For each metric, Friedman across start/max/end, plus pairwise Wilcoxon
    # post-hoc (Holm-corrected over the epoch pairs) so you can see which epochs
    # differ.
    import itertools
    across_metrics = ['bilateral_frac', 'opponent_frac', 'congruent_frac',
                      'ipsi_only_frac', 'contra_only_frac',
                      'opp_frac_bilateral', 'contra_frac_unilateral', 'median_LI']
    across_rows, posthoc_rows = [], []
    for metric in across_metrics:
        piv = per_session.pivot(index='session', columns='epoch',
                                values=metric).reindex(columns=epochs)
        complete = piv.dropna()
        n = int(len(complete))
        if len(epochs) >= 3 and n >= 3:
            try:
                res = stats.friedmanchisquare(*[complete[ep].values for ep in epochs])
                fstat, fp = float(res.statistic), float(res.pvalue)
            except Exception:
                fstat, fp = float('nan'), float('nan')
        else:
            fstat, fp = float('nan'), float('nan')
        across_rows.append({'metric': metric, 'friedman_stat': fstat,
                            'friedman_p': fp, 'n_sessions': n})
        for a, b in itertools.combinations(epochs, 2):
            pair = piv[[a, b]].dropna()
            d = (pair[a].values - pair[b].values)
            d = d[d != 0]
            if d.size >= 1:
                try:
                    pw = float(stats.wilcoxon(d).pvalue)
                except Exception:
                    pw = float('nan')
            else:
                pw = float('nan')
            posthoc_rows.append({'metric': metric, 'epoch_a': a, 'epoch_b': b,
                                 'wilcoxon_p': pw, 'n_sessions': int(len(pair))})
    across_epoch = pd.DataFrame(across_rows)
    across_epoch_posthoc = pd.DataFrame(posthoc_rows)
    # Holm-correct the pairwise tests within each metric
    if not across_epoch_posthoc.empty:
        from statsmodels.stats.multitest import multipletests
        across_epoch_posthoc['wilcoxon_p_holm'] = np.nan
        for _, idx in across_epoch_posthoc.groupby('metric').groups.items():
            sub = across_epoch_posthoc.loc[idx, 'wilcoxon_p']
            finite = sub.notna().values
            if finite.any():
                corr = np.full(len(sub), np.nan)
                corr[finite] = multipletests(sub[sub.notna()].values, method='holm')[1]
                across_epoch_posthoc.loc[idx, 'wilcoxon_p_holm'] = corr

    # ── change across sessions (trend over days) ──────────────────────────────
    # Spearman correlation of each per-session proportion vs session order.
    if session_order is not None:
        order_map = {str(s): i for i, s in enumerate(session_order)}
    else:
        order_map = {s: i for i, s in enumerate(sorted(per_session['session'].unique()))}

    trend_metrics = ['bilateral_frac', 'opponent_frac', 'congruent_frac',
                     'ipsi_only_frac', 'contra_only_frac',
                     'contra_frac_unilateral', 'median_LI']
    trend_rows = []
    for ep in epochs:
        sub = per_session[per_session['epoch'] == ep].copy()
        sub['order'] = sub['session'].map(order_map)
        sub = sub.dropna(subset=['order']).sort_values('order')
        for metric in trend_metrics:
            rho, p, n = _spearman_trend(sub['order'].values, sub[metric].values)
            trend_rows.append({'epoch': ep, 'metric': metric,
                               'spearman_rho': rho, 'p': p, 'n_sessions': n})
    session_trend = pd.DataFrame(trend_rows)

    return {'per_session': per_session, 'within_epoch': within_epoch,
            'across_epoch': across_epoch, 'across_epoch_posthoc': across_epoch_posthoc,
            'session_trend': session_trend}
