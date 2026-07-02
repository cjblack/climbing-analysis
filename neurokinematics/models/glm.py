from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.optimize import minimize
from scipy.special import gammaln  # log gamma for log(y!)
from scipy.signal import decimate, savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold, GroupKFold  # event-grouped CV uses GroupKFold
from scipy import __version__ as scipy_version

from sklearn.preprocessing import StandardScaler
from sklearn import __version__ as sk_version

import statsmodels.api as sm
from statsmodels import __version__ as sm_version

import xarray as xr

from neurokinematics.io import load_zarr, save_model, save_yaml, save_dataset, save_dataframe
from neurokinematics.models.basis import (
    offsets_from_window, raised_cosine_basis, lagged_feature_design,
)
from neurokinematics import __version__ as nk_version


def glm_cv_scores(y, pred, family: str):
    """Held-out goodness-of-fit for a GLM, family-appropriate.

    Args:
        y (np.ndarray): Observed responses (held-out).
        pred (np.ndarray): Out-of-sample predictions, same length as ``y``.
        family (str): statsmodels family name (e.g. ``'Poisson'``, ``'Gaussian'``).

    Returns:
        dict: ``cv_corr`` (Pearson r between observed and predicted) and
        ``cv_r2`` — ordinary R² for Gaussian, deviance-based McFadden-style
        pseudo-R² (``1 - D_model/D_null``) for count families. ``cv_deviance``
        is also included for count families.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[m], pred[m]

    out = {}
    if y.size > 1 and np.std(y) > 0 and np.std(pred) > 0:
        out['cv_corr'] = float(np.corrcoef(y, pred)[0, 1])
    else:
        out['cv_corr'] = float('nan')

    if (family or '').lower() == 'gaussian':
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out['cv_r2'] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    else:
        # Poisson deviance pseudo-R² against an intercept-only (mean-rate) null
        eps = 1e-9
        mu = np.clip(pred, eps, None)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(y > 0, y * np.log(y / mu), 0.0)
        d_model = 2.0 * float(np.sum(term - (y - mu)))
        mu0 = max(y.mean(), eps)
        with np.errstate(divide='ignore', invalid='ignore'):
            term0 = np.where(y > 0, y * np.log(y / mu0), 0.0)
        d_null = 2.0 * float(np.sum(term0 - (y - mu0)))
        out['cv_deviance'] = d_model
        out['cv_r2'] = float(1.0 - d_model / d_null) if d_null > 0 else float('nan')
    return out


def _fit_linear_model(y, X, family: str, alpha: float = 0.0):
    """Fit a GLM, optionally with L2 (ridge) regularization.

    Returns an object exposing ``.predict(X)`` for both paths:
    * ``alpha == 0`` → ``statsmodels`` GLM (unchanged behaviour; also exposes
      ``.aic`` / ``.llf``).
    * ``alpha > 0``  → an L2-penalized scikit-learn estimator (``Ridge`` for
      Gaussian, ``PoissonRegressor`` otherwise). The intercept is fit unpenalized
      (``fit_intercept=True``); the constant column already present in ``X`` is
      mean-centred away and contributes nothing.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if alpha and alpha > 0:
        if str(family).lower() == 'gaussian':
            from sklearn.linear_model import Ridge
            est = Ridge(alpha=float(alpha), fit_intercept=True)
        else:
            from sklearn.linear_model import PoissonRegressor
            est = PoissonRegressor(alpha=float(alpha), fit_intercept=True, max_iter=500)
        est.fit(X, y)
        return est
    return sm.GLM(y, X, family=getattr(sm.families, family)()).fit()


def crossval_glm_predictions(X_model, y, family: str, groups, n_splits: int = 5, alpha: float = 0.0):
    """Event-grouped K-fold out-of-sample GLM predictions.

    Splits by ``groups`` (movement-event id) with :class:`sklearn.model_selection.GroupKFold`
    so whole events are held out together — a bout is never split across
    train/test. Every row is predicted exactly once (from the fold in which its
    event was the test set), so the returned array aligns 1:1 with the input rows.

    Args:
        X_model (pd.DataFrame | np.ndarray): Design matrix including the constant.
        y (np.ndarray): Response, shape ``(n_rows,)``.
        family (str): statsmodels family name.
        groups (np.ndarray): Per-row event id used for grouping.
        n_splits (int): Requested folds; clamped to ``[2, n_unique_groups]``.

    Returns:
        tuple: ``(oos_pred, metrics)`` — held-out predictions (``np.nan`` for any
        fold that failed to converge) and the :func:`glm_cv_scores` dict, with
        ``n_splits`` / ``n_groups`` added.
    """
    X_arr = np.asarray(X_model, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)

    n_groups = int(len(np.unique(groups)))
    n_splits = int(max(2, min(n_splits, n_groups)))

    oos = np.full(y.shape, np.nan, dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(X_arr, y, groups):
        try:
            est = _fit_linear_model(y[train_idx], X_arr[train_idx], family, alpha=alpha)
            oos[test_idx] = est.predict(X_arr[test_idx])
        except Exception:
            # leave NaN for this fold's rows; scoring ignores non-finite entries
            continue

    metrics = glm_cv_scores(y, oos, family)
    metrics['n_splits'] = n_splits
    metrics['n_groups'] = n_groups
    return oos, metrics


def _apply_cv(params, X_model, y, family, groups, insample_pred, alpha: float = 0.0):
    """Run event-grouped CV when ``params['cv']`` requests folds.

    Returns the prediction to report — held-out predictions when CV ran, the
    in-sample prediction otherwise — and records CV metrics on ``params['metrics']``.
    ``alpha`` carries the same L2 penalty used for the full fit into each fold.
    """
    cv_cfg = params.get('cv') or {}
    n_splits = int(cv_cfg.get('n_splits', 0) or 0) if isinstance(cv_cfg, dict) else int(cv_cfg or 0)
    n_groups = int(len(np.unique(groups)))

    metrics = params.setdefault('metrics', {})
    if n_splits >= 2 and n_groups >= 2:
        oos, cv_metrics = crossval_glm_predictions(X_model, y, family, groups, n_splits=n_splits, alpha=alpha)
        metrics.update(cv_metrics)
        metrics['cross_validated'] = True
        return oos
    metrics['cross_validated'] = False
    return insample_pred


def _circular_shift_within_groups(y, groups, rng):
    """Circularly roll the target within each event by a random non-zero offset.

    Breaks the temporal correspondence between spikes and the kinematic while
    preserving each event's marginal distribution and autocorrelation — the basis
    of the trial-shuffle permutation null.
    """
    y = np.asarray(y, dtype=float).copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if idx.size > 1:
            y[idx] = np.roll(y[idx], int(rng.integers(1, idx.size)))
    return y


def shuffle_null_cv_r2(X_model, y, family, groups, n_splits, alpha, real_r2,
                       n_shuffle: int = 100, seed: int = 0):
    """One-sided permutation test for a cross-validated decoder.

    Re-runs event-grouped CV ``n_shuffle`` times with the target circularly
    shifted within each event, building a null distribution of CV R². Returns
    ``(p_value, null_mean)`` where ``p = (#{null >= real} + 1) / (n + 1)``.
    """
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(n_shuffle)):
        y_sh = _circular_shift_within_groups(y, groups, rng)
        _, m = crossval_glm_predictions(X_model, y_sh, family, groups,
                                        n_splits=n_splits, alpha=alpha)
        r2 = m.get('cv_r2')
        if r2 is not None and np.isfinite(r2):
            null.append(r2)
    if not null:
        return float('nan'), float('nan')
    null = np.asarray(null)
    p = (int(np.sum(null >= real_r2)) + 1) / (null.size + 1)
    return float(p), float(null.mean())


def create_glm_encoder(pose_ds: str | Path | xr.Dataset, spike_ds: str | Path | xr.Dataset, params: dict | None = None, save_path: str | Path | None = None):
    """Create glm encoder model from movement and spike data

    Args:
        pose_ds (str | Path | xr.Dataset): Path to or xarray dataset containing binned movement data. If str or Path, then the file must be a zarr store ending in '.zarr'
        spike_ds (str | Path | xr.Dataset): Path to or xarray dataset containing binned spike data. If str or Path, then the file must be a zarr store ending in '.zarr'
        params (dict | None, optional): Dictionary containing parameters for running GLM. Format is:
                params = {
                    'node': str,
                    'type': str,
                    'features': {
                        'pose': list,
                        'spikes': str
                    },
                    'unit': int
                }
            To model how spiking depends on a movement feature across a range of
            temporal offsets (so a unit's lead/lag relative to movement can be
            recovered), supply an optional raised-cosine temporal basis under
            ``params['pose']['basis']``::

                'basis': {
                    'window': (-0.1, 0.2),  # (start_s, end_s) of the kinematic
                                            # sample relative to the spike bin;
                                            # >0 = future (unit leads movement),
                                            # <0 = past (unit lags movement)
                    'n_basis': 5,           # number of raised-cosine bumps
                    'spacing': 'linear',    # 'linear' or 'log'
                }

            When omitted, each feature enters the design at zero lag (same-bin),
            i.e. the original instantaneous behaviour.
        Defaults to None.

    Raises:
        ValueError: Raises when pose_ds is str or Path, and the file format for pose_ds is invalid
        ValueError: Raises when spike_ds is str or Path, and the file format for spike_ds is invalid

    Returns:
        _type_: _description_
    """


    if params is None:
        params = {}

    if isinstance(pose_ds, (str, Path)):
        pose_ds_str = str(pose_ds)
        pose_ds = Path(pose_ds)
        if pose_ds.suffix == '.zarr':
            pose_ds = load_zarr(pose_ds, method='xarray')
        else:
            raise ValueError('Accepted file formats are: ".zarr".')
    else:
        pose_ds_str = params.get('input_data', {}).get('pose_dataset', None)

    if isinstance(spike_ds, (str, Path)):
        spike_ds_str = str(spike_ds)
        spike_ds = Path(spike_ds)
        if spike_ds.suffix == '.zarr':
            spike_ds = load_zarr(spike_ds, method='xarray')
        else:
            raise ValueError('Accepted file formats are: ".zarr".')
    else:
        spike_ds_str = params.get('input_data', {}).get('spike_dataset', None)

    params['input_data'] = {'pose_dataset': pose_ds_str, 'spike_dataset': spike_ds_str}
    node = params.get("pose", {}).get("node", pose_ds.node.values[0]) #glm_params['node']
    family = params.get("family", 'Poisson')
    glm_type = params.get("type", 'encoder')  #glm_params['type']
    pose_feature = params.get("pose", {}).get('features', ['position_y']) #glm_params['features']['pose']
    basis_cfg = params.get("pose", {}).get('basis', None)
    spike_feature = params.get("spikes", {}).get('features', 'spike_counts') #glm_params['features']['spikes']
    spike_feature = spike_feature[0] # this will be a list - but should only contain one entry
    unit = params.get("spikes", {}).get('unit', 0)[0] # this will be a list - but should only contain one entry
    time_bins = spike_ds.time_bin.values

    attrs = {
        "model_type": glm_type,
        "unit": unit,
        "node": node,
        "features": {
            'pose': pose_feature,
            'spikes': spike_feature
        }
    }

    mask = (pose_ds.reference_node == node).compute()
    pose_sub = pose_ds.where(mask, drop=True)
    spike_sub = spike_ds.where(mask, drop=True)

    # pose feature
    #pos = pose_sub.position.sel(node=node)
    predictors = dict()
    features = []
    for pf in pose_feature:
        pf_split = pf.split('_')
        pf_len = len(pf_split)
        feat_name = pf_split[0]
        feat_ = pose_sub[feat_name].sel(node=node)
        if pf_len > 1:
            for fc in pf_split[1]:
                feat_name_ = feat_name+'_'+fc
                feat_data = feat_.sel(coord=fc)
                predictors[feat_name_] = feat_data
                features.append(feat_name_)
        else:
            predictors[feat_name] = feat_
            features.append(feat_name)

    if basis_cfg:
        # Expand each feature onto a raised-cosine temporal basis so the GLM can
        # learn a temporal filter (and thus the unit's lead/lag) rather than a
        # single same-bin coefficient.
        bin_size = float(np.median(np.diff(time_bins)))
        offsets = offsets_from_window(basis_cfg.get('window', (0.0, 0.0)), bin_size)
        basis = raised_cosine_basis(
            offsets,
            n_basis = basis_cfg.get('n_basis', 5),
            spacing = basis_cfg.get('spacing', 'linear'),
            overlap = basis_cfg.get('overlap', 2.0),
        )
        cols = {}
        for name in features:
            design = lagged_feature_design(predictors[name].values, offsets, basis)
            for k in range(design.shape[2]):
                cols[f"{name}__b{k}"] = design[:, :, k].reshape(-1)
        X = pd.DataFrame(cols)
        attrs['basis'] = {
            'window': list(basis_cfg.get('window', (0.0, 0.0))),
            'n_basis': int(basis.shape[1]),
            'spacing': basis_cfg.get('spacing', 'linear'),
            'offsets': offsets.tolist(),
        }
    else:
        X = pd.DataFrame({name: predictors[name].values.reshape(-1) for name in features})

    spikes = spike_sub[spike_feature].sel(unit=unit)#.isel(time_bin=slice(1, None))
    n_events, n_bins = spikes.shape

    event_idx = np.repeat(np.arange(n_events), n_bins)
    time_idx = np.tile(np.arange(n_bins), n_events)

    valid = (
        pose_sub.valid.fillna(False).astype(bool) &#.isel(time_bin = slice(1, None)) &
        spike_sub.valid.fillna(False).astype(bool)#.isel(time_bin = slice(1, None))
    )

    sy = spikes.values.reshape(-1)

    valid_flat = valid.values.reshape(-1)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(sy)

    keep = valid_flat & finite
    X = X.loc[keep]
    sy = sy[keep]
    event_idx = event_idx[keep]
    time_idx = time_idx[keep]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns, index = X.index)
    X_model = sm.add_constant(X_scaled, has_constant="add")
    model = sm.GLM(sy, X_model, family=getattr(sm.families, family)())

    results = model.fit()

    insample_pred = results.predict(X_model)

    params['packages'] = {'statsmodels': sm_version, 'scipy': scipy_version, 'sklearn': sk_version, 'neurokinematics': nk_version}
    params['metrics'] = {'aic': float(results.aic), 'log_likelihood': float(results.llf)}

    # event-grouped cross-validation (held-out predictions) when params['cv'] is set
    predicted = _apply_cv(params, X_model, sy, family, event_idx, insample_pred)

    outputs = {
        'predicted': predicted,
        'observed': sy,
        'event_idx': event_idx,
        'time_idx': time_idx,
        'time_bins':time_bins,
        'attrs': attrs,
        'params': params
        }

    if save_path:
        save_path = Path(save_path)
        created_on = datetime.now().strftime('%Y%m%d_%H_%M_%S') # get creation date
        save_path = save_path / 'glm' / glm_type / f'{node}_to_unit_{unit}_{created_on}'
        save_glm_results(model, results, outputs, params, save_path)


    return model, results, outputs

def create_glm_decoder(pose_ds: str | Path | xr.Dataset, spike_ds: str | Path | xr.Dataset, params: dict | None = None, save_path: str | Path | None = None):
    """Create glm decoder model from movement and spike data

    Args:
        pose_ds (str | Path | xr.Dataset): Path to or xarray dataset containing binned movement data. If str or Path, then the file must be a zarr store ending in '.zarr'
        spike_ds (str | Path | xr.Dataset): Path to or xarray dataset containing binned spike data. If str or Path, then the file must be a zarr store ending in '.zarr'
        params (dict | None, optional): Dictionary containing parameters for running GLM. Format is:
                params = {
                    'node': str,
                    'type': str,
                    'features': {
                        'pose': list,
                        'spikes': str
                    },
                    'unit': int
                }
        Defaults to None.

    Raises:
        ValueError: Raises when pose_ds is str or Path, and the file format for pose_ds is invalid
        ValueError: Raises when spike_ds is str or Path, and the file format for spike_ds is invalid

    Returns:
        _type_: _description_
    """


    if params is None:
        params = {}

    if isinstance(pose_ds, (str, Path)):
        pose_ds_str = str(pose_ds)
        pose_ds = Path(pose_ds)
        if pose_ds.suffix == '.zarr':
            pose_ds = load_zarr(pose_ds, method='xarray')
        else:
            raise ValueError('Accepted file formats are: ".zarr".')
    else:
        pose_ds_str = params.get('input_data', {}).get('pose_dataset', None)

    if isinstance(spike_ds, (str, Path)):
        spike_ds_str = str(spike_ds)
        spike_ds = Path(spike_ds)
        if spike_ds.suffix == '.zarr':
            spike_ds = load_zarr(spike_ds, method='xarray')
        else:
            raise ValueError('Accepted file formats are: ".zarr".')
    else:
        spike_ds_str = params.get('input_data', {}).get('spike_dataset', None)

    params['input_data'] = {'pose_dataset': pose_ds_str, 'spike_dataset': spike_ds_str}
    node = params.get("pose", {}).get("node", pose_ds.node.values[0]) #glm_params['node']
    family = params.get("family", 'Gaussian') # set gaussian by default for decoding
    glm_type = params.get("type", 'decoder')  #glm_params['type']
    target = params.get("pose", {}).get('features', ['position_y'])[0] #glm_params['features']['pose']
    # directional targets are 'feature_coord' (e.g. 'velocity_y'); scalars like
    # 'speed' have no coord component
    if '_' in target:
        pose_feature, pose_coord = target.split('_', 1)
    else:
        pose_feature, pose_coord = target, None
    spike_feature = params.get("spikes", {}).get('features', 'spike_counts') #glm_params['features']['spikes']
    spike_feature = spike_feature[0] # this will be a list - but should only contain one entry
    units = params.get("spikes", {}).get('unit', 0) # this will be a list - but should only contain one entry
    time_bins = spike_ds.time_bin.values

    attrs = {
        "model_type": glm_type,
        "unit": units,
        "node": node,
        "target": target,
        "features": {
            'pose': pose_feature,
            'spikes': spike_feature
        }
    }

    all_events = bool(params.get("pose", {}).get('all_events', False))
    if all_events:
        # decode this node's kinematics from *every* movement bout, not only the
        # bouts it initiated — more (and more varied) training data
        pose_sub, spike_sub = pose_ds, spike_ds
    else:
        mask = (pose_ds.reference_node == node).compute()
        pose_sub = pose_ds.where(mask, drop=True)
        spike_sub = spike_ds.where(mask, drop=True)

    # population predictors (one column per unit, or per unit × lag-basis)
    predictors = dict()
    features = []
    spike_basis = params.get("spikes", {}).get('basis', None)
    bin_size = float(np.median(np.diff(time_bins)))

    # optional Gaussian smoothing of each unit's binned spikes -> firing rate.
    # Raw counts at small bins are 0/1-sparse and very noisy; smoothing denoises
    # the predictors and usually improves decoding a lot.
    smoothing_s = float(params.get("spikes", {}).get('smoothing_s', 0.0) or 0.0)
    sigma_bins = (smoothing_s / bin_size) if smoothing_s > 0 else 0.0

    def _unit_series(uid):
        arr = spike_sub[spike_feature].isel(unit=uid).values.astype(float)  # (event, bins)
        if sigma_bins > 0:
            # smooth within each event (axis=1); 'constant' avoids cross-event bleed
            arr = gaussian_filter1d(arr, sigma=sigma_bins, axis=1, mode='constant')
        return arr

    n_events, n_bins = spike_sub[spike_feature].isel(unit=units[0]).shape

    if spike_basis:
        offsets = offsets_from_window(spike_basis.get('window', (0.0, 0.0)), bin_size)
        basis = raised_cosine_basis(
            offsets,
            n_basis=spike_basis.get('n_basis', 5),
            spacing=spike_basis.get('spacing', 'linear'),
        )
        cols = {}
        for uid in units:
            design = lagged_feature_design(_unit_series(uid), offsets, basis)
            for k in range(design.shape[2]):
                name = f"unit_{uid}__b{k}"
                cols[name] = design[:, :, k].reshape(-1)
                features.append(name)
        X = pd.DataFrame(cols)
        attrs['spike_basis'] = {
            'window': list(spike_basis.get('window', (0.0, 0.0))),
            'n_basis': int(basis.shape[1]),
            'spacing': spike_basis.get('spacing', 'linear'),
            'offsets': offsets.tolist(),
        }
    else:
        for uid in units:
            name = f"unit_{uid}"
            predictors[name] = _unit_series(uid)
            features.append(name)
        X = pd.DataFrame({name: predictors[name].reshape(-1) for name in features})

    attrs['all_events'] = all_events
    attrs['smoothing_s'] = smoothing_s

    event_idx = np.repeat(np.arange(n_events), n_bins)
    time_idx = np.tile(np.arange(n_bins), n_events)

    valid = (
        pose_sub.valid.fillna(False).astype(bool) &#.isel(time_bin = slice(1, None)) &
        spike_sub.valid.fillna(False).astype(bool)#.isel(time_bin = slice(1, None))
    )

    target_da = pose_sub[pose_feature].sel(node=node)
    if pose_coord is not None:
        target_da = target_da.sel(coord=pose_coord)
    sy = target_da.values.reshape(-1)

    # optional co-movement control: regress the same feature on another limb out of
    # the target, so we decode the part of (e.g.) ipsilateral speed that is NOT
    # shared with the contralateral limb — the residual.
    partial_node = params.get("pose", {}).get("partial_out_node")
    cy = None
    if partial_node:
        cov_da = pose_sub[pose_feature].sel(node=partial_node)
        if pose_coord is not None:
            cov_da = cov_da.sel(coord=pose_coord)
        cy = cov_da.values.reshape(-1)

    valid_flat = valid.values.reshape(-1)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(sy)
    if cy is not None:
        finite = finite & np.isfinite(cy)

    keep = valid_flat & finite
    X = X.loc[keep]
    sy = sy[keep]
    event_idx = event_idx[keep]
    time_idx = time_idx[keep]

    if cy is not None:
        # residualise sy on [1, cy]: remove the linear component shared with the
        # partial-out limb; decode only the orthogonal (independent) part.
        cyk = cy[keep]
        A = np.column_stack([np.ones_like(cyk), cyk])
        beta, *_ = np.linalg.lstsq(A, sy, rcond=None)
        sy = sy - A @ beta
        attrs['partial_out_node'] = str(partial_node)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns = X.columns, index = X.index)
    X_model = sm.add_constant(X_scaled, has_constant="add")

    # optional L2 (ridge) regularization — important once spike-history lags make
    # the design wide and collinear
    reg = params.get('regularization') or {}
    alpha = float(reg.get('alpha', 0.0) or 0.0) if isinstance(reg, dict) else float(reg or 0.0)

    fitted = _fit_linear_model(sy, X_model, family, alpha=alpha)
    model = results = fitted
    insample_pred = np.asarray(fitted.predict(X_model))

    params['packages'] = {'statsmodels': sm_version, 'scipy': scipy_version, 'sklearn': sk_version, 'neurokinematics': nk_version}
    params['metrics'] = {} if alpha > 0 else {'aic': float(results.aic), 'log_likelihood': float(results.llf)}
    if alpha > 0:
        params['metrics']['regularization_alpha'] = alpha

    # event-grouped cross-validation (held-out predictions) when params['cv'] is set
    predicted = _apply_cv(params, X_model, sy, family, event_idx, insample_pred, alpha=alpha)

    # optional permutation null: is the CV R² above chance? (trial-shuffle)
    shuf = params.get('shuffle') or {}
    n_shuffle = int(shuf.get('n', 0) or 0) if isinstance(shuf, dict) else int(shuf or 0)
    real_r2 = params['metrics'].get('cv_r2')
    if n_shuffle and params['metrics'].get('cross_validated') and real_r2 is not None and np.isfinite(real_r2):
        n_splits_used = int(params['metrics'].get('n_splits', 5))
        p, null_mean = shuffle_null_cv_r2(
            X_model, sy, family, event_idx, n_splits_used, alpha, real_r2, n_shuffle=n_shuffle)
        params['metrics']['shuffle_p'] = p
        params['metrics']['shuffle_null_mean'] = null_mean
        params['metrics']['shuffle_n'] = n_shuffle

    outputs = {
        'predicted': predicted,
        'observed': sy,
        'event_idx': event_idx,
        'time_idx': time_idx,
        'time_bins':time_bins,
        'attrs': attrs,
        'params': params
        }

    if save_path:
        save_path = Path(save_path)
        created_on = datetime.now().strftime('%Y%m%d_%H_%M_%S') # get creation date
        save_path = save_path / 'glm' / glm_type / f'population_to_{node}_{target}_{created_on}'
        save_glm_results(model, results, outputs, params, save_path)
        

    return model, results, outputs

def save_glm_results(model, results, outputs, params, save_path):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    model_save_path = save_path / 'glm_model.joblib'
    params_save_path = save_path / 'glm_params.yaml'
    save_model(model, model_save_path, method = 'joblib')
    save_yaml(params, params_save_path)
    _ = build_glm_dataset(outputs, attrs=outputs['attrs'], save_path=save_path)


def build_encoder_params(node, unit, features, family: str = "Poisson",
                         mode: str = "full", basis: dict | None = None,
                         n_splits: int = 0):
    """Assemble a params dict for :func:`create_glm_encoder` / :func:`compare_glm_models`.

    Pure helper (no GUI dependency) so the parameter spec can be unit-tested and
    reused by the GUI encoder dialog.

    Args:
        node (str): Reference node to model (e.g. ``'hand'``).
        unit (int | list): Unit id, or list of unit ids. Coerced to a list.
        features (list): Pose feature names, e.g. ``['velocity_x', 'velocity_y']``.
        family (str, optional): GLM family name on ``statsmodels.families``.
            Defaults to ``'Poisson'``.
        mode (str, optional): Comparison mode passed to
            :func:`build_glm_model_sets` (``'single'``, ``'full'``,
            ``'single_and_full'``, ``'drop_one'``). Defaults to ``'full'``.
        basis (dict | None, optional): Raised-cosine temporal-basis spec with keys
            ``window`` (``(start_s, end_s)``), ``n_basis``, and ``spacing``. When
            None, features enter the design at zero lag (same-bin). Defaults to None.
        n_splits (int, optional): Event-grouped CV folds. ``0`` (default) fits/scores
            in-sample; ``>=2`` adds ``params['cv']`` so the encoder reports held-out
            predictions and a cross-validated pseudo-R².

    Returns:
        dict: A params dict consumable by the encoder functions.
    """
    units = list(unit) if isinstance(unit, (list, tuple)) else [unit]
    params = {
        "type": "encoder",
        "family": family,
        "pose": {"node": node, "features": list(features)},
        "spikes": {"unit": units, "features": ["spike_counts"]},
        "comparison": {"mode": mode},
    }
    if basis:
        params["pose"]["basis"] = {
            "window": list(basis.get("window", (0.0, 0.0))),
            "n_basis": int(basis.get("n_basis", 5)),
            "spacing": basis.get("spacing", "linear"),
        }
    if n_splits:
        params["cv"] = {"n_splits": int(n_splits)}
    return params


def build_decoder_params(node, units, target, family: str = "Gaussian",
                         n_splits: int = 5, lag: dict | None = None, alpha: float = 0.0,
                         smoothing_s: float = 0.0, all_events: bool = False,
                         n_shuffle: int = 0, partial_out_node: str | None = None):
    """Assemble a params dict for :func:`create_glm_decoder`.

    A decoder regresses a movement feature on a *population* of units — i.e. "can
    these neurons reconstruct speed / position?". Pure helper (no GUI dependency)
    so it can be unit-tested and reused by the GUI decoder dialog.

    Args:
        node (str): Reference node whose movement is decoded.
        units (list): Population of unit ids used as predictors.
        target (str): Movement feature to decode — a scalar like ``'speed'`` or a
            directional ``'feature_coord'`` like ``'position_y'`` / ``'velocity_x'``.
        family (str, optional): GLM family. Defaults to ``'Gaussian'`` (continuous
            kinematics).
        n_splits (int, optional): Event-grouped CV folds. ``0`` fits in-sample;
            ``>=2`` (default 5) reports held-out predictions and a CV R².
        lag (dict | None, optional): Spike-history window — each unit's spikes are
            expanded onto a raised-cosine lag basis (keys ``window`` ``(start_s,
            end_s)``, ``n_basis``, ``spacing``). None = same-bin counts only.
        alpha (float, optional): L2 (ridge) penalty. ``0`` = ordinary least
            squares; ``>0`` regularizes — recommended once lags widen the design.
        smoothing_s (float, optional): Gaussian smoothing σ (seconds) applied to
            each unit's binned spikes (→ firing rate) before lagging. ``0`` = raw
            counts. Denoising the predictors usually helps decoding.
        all_events (bool, optional): If True, decode the node's kinematics from
            *all* movement events rather than only the ones it initiated
            (``reference_node == node``) — more, more varied training data.
        n_shuffle (int, optional): If ``>0`` and CV is on, run a trial-shuffle
            permutation test and record a p-value for the CV R². ``0`` = skip.

    Returns:
        dict: A params dict consumable by :func:`create_glm_decoder`.
    """
    params = {
        "type": "decoder",
        "family": family,
        "pose": {"node": node, "features": [target]},
        "spikes": {"unit": list(units), "features": ["spike_counts"]},
    }
    if lag:
        params["spikes"]["basis"] = {
            "window": list(lag.get("window", (-0.15, 0.15))),
            "n_basis": int(lag.get("n_basis", 5)),
            "spacing": lag.get("spacing", "linear"),
        }
    if smoothing_s and smoothing_s > 0:
        params["spikes"]["smoothing_s"] = float(smoothing_s)
    if all_events:
        params["pose"]["all_events"] = True
    if partial_out_node:
        params["pose"]["partial_out_node"] = str(partial_out_node)
    if alpha and alpha > 0:
        params["regularization"] = {"alpha": float(alpha)}
    if n_shuffle and n_shuffle > 0:
        params["shuffle"] = {"n": int(n_shuffle)}
    if n_splits:
        params["cv"] = {"n_splits": int(n_splits)}
    return params


def build_glm_model_sets(features, mode: str = "full"):
    if mode == "single_and_full":
        model_sets = {
            feat: [feat] for feat in features
        }
        model_sets['full'] = features

    elif mode == "single":
        model_sets = {
            feat: [feat] for feat in features
        }
    elif mode == 'full':
        model_sets = {
            'full': features
        }
    elif mode == 'drop_one':
        model_sets = {
            'full': features
        }
        for feat in features:
            model_sets[f"drop_{feat}"] = [
                f for f in features if f != feat
            ]
    else:
        raise ValueError(
            "mode must be one of: " \
            "'single', 'full', 'single_and_full', 'drop_one'"
        )
    
    return model_sets


def compare_glm_models(x_ds, y_ds, params, save_path):

    created_on = datetime.now().strftime('%Y%m%d_%H_%M_%S')

    if isinstance(x_ds, (str, Path)):
        x_ds_str = str(x_ds)
        x_ds = Path(x_ds)
        if x_ds.suffix == '.zarr':
            x_ds = load_zarr(x_ds, method='xarray')
        else:
            raise ValueError('Accepted file formats are: ".zarr".')
    else:
        x_ds_str = None
    
    if isinstance(y_ds, (str, Path)):
        y_ds_str = str(y_ds)
        y_ds = Path(y_ds)
        if y_ds.suffix == '.zarr':
            y_ds = load_zarr(y_ds, method='xarray')
        else:
            raise ValueError('Accepted file formats are: ".zarr".')
    else:
        y_ds_str = None



    params['input_data'] = {'x_dataset': x_ds_str, 'y_dataset': y_ds_str}
    glm_type = params['type']
    mode = params.get('comparison', {}).get('mode', 'full')

    if glm_type == 'encoder':
        node = params.get('pose', {}).get('node', x_ds.node.values[0])
        unit = params.get('spikes', {}).get('unit', 0)
        features = params.get('pose', {}).get('features', ['position_x'])
        glm_save_directory = f'comparison_{mode}_{node}_unit_{unit}_{created_on}'
    else:
        raise NotImplementedError(f"compare_glm_models only supports glm_type='encoder', got '{glm_type}'.")

    model_sets = build_glm_model_sets(features, mode=mode)

    if save_path:
        save_path = Path(save_path) / 'glm' / glm_type / glm_save_directory 

    fitted_models = {}
    summary_rows = []

    for model_name, feature_set in tqdm(model_sets.items(), total=len(model_sets), desc="Fitting models", unit="models"):
        params_ = deepcopy(params)
        params_['pose']['features'] = feature_set
        params_['comparison_model'] = model_name

        model, results, outputs = create_glm_encoder(
            x_ds,
            y_ds,
            params = params_,
        )
        fitted_models[model_name] = {
            'model': model,
            'results': results,
            'outputs': outputs,
            'params': params_
        }
        if save_path:
            save_glm_results(model, results, outputs, params_, save_path / model_name)
            
        summary_rows.append(
            {
                'model_name': model_name,
                'features': ", ".join(feature_set),
                'aic': float(results.aic),
                'log_likelihood': float(results.llf),

            }
        )

    summary = pd.DataFrame(summary_rows)
    if save_path:
        save_dataframe(summary, save_path / 'summary.csv', storage_format = 'csv')

    
    return fitted_models, summary


def plot_glm_results(spikes_binned, predicted_rate, bin_width):
    time = np.arange(len(spikes_binned)) * bin_width

    # Null model prediction (mean spike count per bin)
    null_rate = np.full_like(spikes_binned, fill_value=np.mean(spikes_binned))

    # Optional: Smooth for visual clarity
    from scipy.signal import savgol_filter

    smooth_actual = savgol_filter(spikes_binned, window_length=11, polyorder=2)
    smooth_pred = savgol_filter(predicted_rate, window_length=11, polyorder=2)
    smooth_null = savgol_filter(null_rate, window_length=11, polyorder=2)
    # Calculate MSE
    mse = np.mean((predicted_rate-spikes_binned)**2)
    print(f'MSE: {mse}')
    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(time, smooth_actual, label="Actual", color='black', linewidth=1)
    plt.plot(time, smooth_pred, label="GLM Predicted", color='red', linestyle='--')
    plt.plot(time, smooth_null, label="Null Model", color='blue', linestyle=':')

    plt.xlabel("Time (s)")
    plt.ylabel("Spike count (smoothed)")
    plt.title(f"Actual vs. GLM vs. Null Model Prediction, MSE: {mse}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def build_glm_dataset(outputs:dict, event_ids = None, attrs = None, save_path: str | Path | None = None):
    y = np.asarray(outputs['observed'])
    pred = np.asarray(outputs['predicted'])
    event_idx = np.asarray(outputs['event_idx'])
    time_idx = np.asarray(outputs['time_idx'])
    time_bins = np.asarray(outputs['time_bins'])
    #print(f'original time_bins: {len(time_bins)}')

    n_events = int(event_idx.max()) + 1
    n_time = len(time_bins)
    #n_time = int(time_idx.max()) + 1

    observed = np.full((n_events, n_time), np.nan)
    predicted = np.full((n_events, n_time), np.nan)
    valid = np.zeros((n_events, n_time), dtype=bool)

    observed[event_idx, time_idx] = y
    predicted[event_idx, time_idx] = pred
    valid[event_idx, time_idx] = True

    if event_ids is None:
        event_ids = np.arange(n_events)
    
    ds = xr.Dataset(
        data_vars = {
            'observed_counts':(
                ['event', 'time_bin'],
                observed
            ),
            'predicted_counts': (
                ['event', 'time_bin'],
                predicted
            ),
            'residuals':(
                ['event', 'time_bin'],
                observed - predicted
            ),
            'valid': (
                ['event', 'time_bin'],
                valid
            )
        },
        coords = {
            'event':event_ids,
            'time_bin': time_bins
        },
        attrs = attrs or {}

    )

    if save_path:
        save_path = Path(save_path) / 'predictions.zarr'
        save_dataset(
            ds,
            save_path,
            chunks = {
                'event': min(100, n_events),
                'time_bin': -1
            }
        )

    return ds


# ── Multi-limb encoder: do limbs contribute *uniquely* (vs co-movement)? ──────

def build_multilimb_encoder_params(nodes, unit, features, family: str = "Poisson",
                                   n_splits: int = 5, basis: dict | None = None,
                                   alpha: float = 0.0):
    """Params for :func:`create_multilimb_encoder` / :func:`compare_limb_contributions`.

    A multi-limb encoder predicts one unit's spiking from the kinematics of **all**
    ``nodes`` simultaneously. Dropping a limb and comparing cross-validated fit
    (see :func:`compare_limb_contributions`) tests whether that limb adds *unique*
    predictive power — i.e. whether apparent bilateral tuning survives controlling
    for the co-movement of the other limbs.

    Args:
        nodes (list): Limbs whose kinematics enter the design (e.g. all four paws).
        unit (int | list): Unit id to model.
        features (list): Pose features per limb, e.g. ``['velocity_x','velocity_y']``.
        family (str): GLM family (default ``'Poisson'``).
        n_splits (int): Event-grouped CV folds (default 5; needed for the drop-one
            comparison to be meaningful).
        basis (dict | None): Optional raised-cosine temporal basis (``window``,
            ``n_basis``, ``spacing``).
        alpha (float): L2 (ridge) penalty; recommended once the design is wide.

    Returns:
        dict: params consumable by the multi-limb encoder functions.
    """
    units = list(unit) if isinstance(unit, (list, tuple)) else [unit]
    params = {
        "type": "multilimb_encoder",
        "family": family,
        "pose": {"nodes": list(nodes), "features": list(features)},
        "spikes": {"unit": units, "features": ["spike_counts"]},
    }
    if basis:
        params["pose"]["basis"] = {
            "window": list(basis.get("window", (0.0, 0.0))),
            "n_basis": int(basis.get("n_basis", 5)),
            "spacing": basis.get("spacing", "linear"),
        }
    if alpha and alpha > 0:
        params["regularization"] = {"alpha": float(alpha)}
    if n_splits:
        params["cv"] = {"n_splits": int(n_splits)}
    return params


def _multilimb_feature_array(ds, node, feat):
    """Per-event feature array (event, time_bin) for one node, parsing 'feat_coord'."""
    if '_' in feat:
        name, coord = feat.split('_', 1)
        da = ds[name].sel(node=node)
        if 'coord' in da.dims:
            da = da.sel(coord=coord)
        return da
    return ds[feat].sel(node=node)


def _build_multilimb_design(pose_sub, nodes, features, basis_cfg, bin_size):
    """Design matrix with one column per (limb, feature) — or (limb, feature, basis).

    Returns ``(X, limb_cols, basis_meta)`` where ``limb_cols[node]`` lists that
    limb's columns (used to drop a limb for the unique-contribution comparison).
    """
    cols, limb_cols, basis_meta = {}, {str(n): [] for n in nodes}, None
    if basis_cfg:
        offsets = offsets_from_window(basis_cfg.get('window', (0.0, 0.0)), bin_size)
        basis = raised_cosine_basis(offsets, n_basis=basis_cfg.get('n_basis', 5),
                                    spacing=basis_cfg.get('spacing', 'linear'))
        for n in nodes:
            for feat in features:
                arr = _multilimb_feature_array(pose_sub, n, feat).values
                design = lagged_feature_design(arr, offsets, basis)
                for k in range(design.shape[2]):
                    c = f"{n}__{feat}__b{k}"
                    cols[c] = design[:, :, k].reshape(-1)
                    limb_cols[str(n)].append(c)
        basis_meta = {'window': list(basis_cfg.get('window', (0.0, 0.0))),
                      'n_basis': int(basis.shape[1]),
                      'spacing': basis_cfg.get('spacing', 'linear')}
    else:
        for n in nodes:
            for feat in features:
                arr = _multilimb_feature_array(pose_sub, n, feat).values
                c = f"{n}__{feat}"
                cols[c] = arr.reshape(-1)
                limb_cols[str(n)].append(c)
    return pd.DataFrame(cols), limb_cols, basis_meta


def create_multilimb_encoder(pose_ds, spike_ds, params: dict | None = None,
                             save_path: str | Path | None = None):
    """Encode one unit's spikes from the kinematics of *all* limbs jointly.

    Unlike :func:`create_glm_encoder` (which masks to one reference limb's events),
    this uses every movement event and stacks per-limb feature columns, so the fit
    partials out the shared (co-)movement: a limb's coefficients reflect its
    contribution *beyond* the other limbs. Returns ``(model, results, outputs)``.
    """
    params = params or {}
    if isinstance(pose_ds, (str, Path)):
        pose_ds = Path(pose_ds)
        if pose_ds.suffix != '.zarr':
            raise ValueError('Accepted file formats are: ".zarr".')
        pose_ds = load_zarr(pose_ds, method='xarray')
    if isinstance(spike_ds, (str, Path)):
        spike_ds = Path(spike_ds)
        if spike_ds.suffix != '.zarr':
            raise ValueError('Accepted file formats are: ".zarr".')
        spike_ds = load_zarr(spike_ds, method='xarray')

    nodes = params.get("pose", {}).get("nodes", [str(n) for n in pose_ds.node.values])
    features = params.get("pose", {}).get("features", ["velocity_x", "velocity_y"])
    basis_cfg = params.get("pose", {}).get("basis", None)
    family = params.get("family", "Poisson")
    spike_feature = params.get("spikes", {}).get("features", ["spike_counts"])[0]
    unit = params.get("spikes", {}).get("unit", [0])
    unit = unit[0] if isinstance(unit, (list, tuple)) else unit
    time_bins = spike_ds.time_bin.values
    bin_size = float(np.median(np.diff(time_bins)))

    reg = params.get('regularization') or {}
    alpha = float(reg.get('alpha', 0.0) or 0.0) if isinstance(reg, dict) else float(reg or 0.0)

    X, limb_cols, basis_meta = _build_multilimb_design(pose_ds, nodes, features, basis_cfg, bin_size)

    spikes = spike_ds[spike_feature].sel(unit=unit)
    n_events, n_bins = spikes.shape
    event_idx = np.repeat(np.arange(n_events), n_bins)
    time_idx = np.tile(np.arange(n_bins), n_events)

    valid = (pose_ds.valid.fillna(False).astype(bool) & spike_ds.valid.fillna(False).astype(bool))
    sy = spikes.values.reshape(-1)
    valid_flat = valid.values.reshape(-1)
    finite = np.isfinite(X).all(axis=1).values & np.isfinite(sy)
    keep = valid_flat & finite

    X = X.loc[keep]
    sy = sy[keep]
    event_idx = event_idx[keep]
    time_idx = time_idx[keep]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_model = sm.add_constant(X_scaled, has_constant="add")

    if alpha and alpha > 0:
        fitted = _fit_linear_model(sy, X_model, family, alpha=alpha)
    else:
        # fit statsmodels directly on the DataFrame so coefficients keep their
        # (limb, feature) names
        fitted = sm.GLM(sy, X_model, family=getattr(sm.families, family)()).fit()
    model = results = fitted
    insample_pred = np.asarray(fitted.predict(X_model))

    params['packages'] = {'statsmodels': sm_version, 'scipy': scipy_version,
                          'sklearn': sk_version, 'neurokinematics': nk_version}
    params['metrics'] = {} if alpha > 0 else {'aic': float(results.aic),
                                              'log_likelihood': float(results.llf)}
    if alpha > 0:
        params['metrics']['regularization_alpha'] = alpha
    predicted = _apply_cv(params, X_model, sy, family, event_idx, insample_pred, alpha=alpha)

    attrs = {
        "model_type": "multilimb_encoder", "unit": unit, "nodes": list(nodes),
        "features": {"pose": list(features), "spikes": spike_feature},
        "limb_columns": {k: v for k, v in limb_cols.items()},
    }
    if basis_meta:
        attrs["basis"] = basis_meta
    outputs = {'predicted': predicted, 'observed': sy, 'event_idx': event_idx,
               'time_idx': time_idx, 'time_bins': time_bins, 'attrs': attrs, 'params': params}

    if save_path:
        save_path = Path(save_path)
        created_on = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        save_path = save_path / 'glm' / 'multilimb_encoder' / f'unit_{unit}_{created_on}'
        save_glm_results(model, results, outputs, params, save_path)

    return model, results, outputs


def compare_limb_contributions(pose_ds, spike_ds, params, save_path=None):
    """Unique cross-validated contribution of each limb (full vs drop-one-limb).

    Fits the full multi-limb encoder, then refits dropping each limb in turn;
    ``unique_cv_r2 = cv_r2(full) - cv_r2(drop_limb)`` is how much predictive power
    that limb adds beyond the others. A limb whose apparent tuning is only
    co-movement collapses to ~0; a genuinely encoded limb stays positive.

    Returns ``(fitted, summary)`` — fitted models per set and a summary DataFrame
    (``model``, ``dropped``, ``limbs``, ``cv_r2``, ``unique_cv_r2``).
    """
    if isinstance(pose_ds, (str, Path)):
        pose_ds = load_zarr(Path(pose_ds), method='xarray')
    if isinstance(spike_ds, (str, Path)):
        spike_ds = load_zarr(Path(spike_ds), method='xarray')

    nodes = list(params.get("pose", {}).get("nodes", [str(n) for n in pose_ds.node.values]))
    fitted, rows = {}, []

    _, _, full = create_multilimb_encoder(pose_ds, spike_ds, deepcopy(params))
    full_r2 = full['params'].get('metrics', {}).get('cv_r2')
    fitted['full'] = full
    rows.append({'model': 'full', 'dropped': None, 'limbs': ", ".join(map(str, nodes)),
                 'cv_r2': full_r2, 'unique_cv_r2': float('nan')})

    for n in nodes:
        keep_nodes = [x for x in nodes if x != n]
        if not keep_nodes:
            continue
        p = deepcopy(params)
        p['pose']['nodes'] = keep_nodes
        _, _, out = create_multilimb_encoder(pose_ds, spike_ds, p)
        r2 = out['params'].get('metrics', {}).get('cv_r2')
        fitted[f'drop_{n}'] = out
        unique = (full_r2 - r2) if (full_r2 is not None and r2 is not None
                                    and np.isfinite(full_r2) and np.isfinite(r2)) else float('nan')
        rows.append({'model': f'drop_{n}', 'dropped': str(n),
                     'limbs': ", ".join(map(str, keep_nodes)), 'cv_r2': r2,
                     'unique_cv_r2': unique})

    summary = pd.DataFrame(rows)
    if save_path:
        save_path = Path(save_path)
        created_on = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        out_dir = save_path / 'glm' / 'multilimb_encoder' / f'limb_contributions_unit_{params.get("spikes",{}).get("unit",["?"])[0]}_{created_on}'
        out_dir.mkdir(parents=True, exist_ok=True)
        save_dataframe(summary, out_dir / 'limb_contributions.csv', storage_format='csv')

    return fitted, summary


# ── Across-session population decoding (with co-movement control) ─────────────

def _find_binned_stores(pose_dir, spikes_dir):
    """``{bin_ms: (pose_zarr, spike_zarr)}`` for bins present in both folders."""
    import re
    out_pose, out_spk = {}, {}
    for folder, store, pat in ((pose_dir, out_pose, "resampled_movements_*ms.zarr"),
                               (spikes_dir, out_spk, "movement_spike_counts_*ms.zarr")):
        if folder and Path(folder).exists():
            for p in Path(folder).glob(pat):
                m = re.search(r"_(\d+)ms\.zarr$", p.name)
                if m:
                    store[int(m.group(1))] = p
    return {ms: (out_pose[ms], out_spk[ms]) for ms in sorted(set(out_pose) & set(out_spk))}


def _session_good_units(spikes_dir):
    """Good unit ids from a session's phy cluster_group.tsv, or None."""
    if not spikes_dir:
        return None
    hits = list(Path(spikes_dir).glob("*/phy_output/cluster_group.tsv"))
    if not hits:
        return None
    try:
        from neurokinematics.ephys.spikes.curation import good_unit_ids
        return good_unit_ids(hits[0].parent)
    except Exception:
        return None


def _iter_decode_sessions(sessions, bin_ms=None):
    """Normalise ``sessions`` to ``[(label, pose_zarr, spike_zarr, spikes_dir)]``.

    Accepts a ``{label: (pose_zarr, spike_zarr[, spikes_dir])}`` mapping, an
    ``ExperimentSubject``-like object (``.sessions``), or an iterable of
    session-like objects (``.dirs``, ``.session_id``).
    """
    def _from_obj(sess):
        dirs = getattr(sess, 'dirs', {})
        stores = _find_binned_stores(dirs.get('pose'), dirs.get('spikes'))
        if not stores:
            return None
        if bin_ms is not None:
            if bin_ms not in stores:
                return None                       # honour the requested bin strictly
            ms = bin_ms
        else:
            ms = max(stores)                      # default: largest (least noisy) bin
        pose_p, spike_p = stores[ms]
        return (str(getattr(sess, 'session_id', sess)), pose_p, spike_p, dirs.get('spikes'))

    if isinstance(sessions, dict):
        out = []
        for label, val in sessions.items():
            sdir = val[2] if len(val) >= 3 else None
            out.append((str(label), val[0], val[1], sdir))
        return out
    seq = sessions.sessions if hasattr(sessions, 'sessions') else sessions
    return [r for r in (_from_obj(s) for s in (seq or [])) if r]


def decode_across_sessions(sessions, node, target="speed", contra_node=None,
                           good_units="auto", select=None, bin_ms=None, n_splits=5,
                           n_shuffle=100, smoothing_s=0.05, lag=None, alpha=1.0,
                           all_events=True, save_path=None):
    """Per-session population decode of a kinematic, summarised across sessions.

    For each session this locates the binned pose/spike zarr stores, resolves the
    decoding population (good units by default), and decodes ``target`` for:
    ``ipsi`` (``node``); and — if ``contra_node`` is given — ``contra``
    (``contra_node``) and ``ipsi_residual`` (``node`` after regressing out
    ``contra_node``, the co-movement control). Each decode is event-grouped
    cross-validated with a shuffle null. Sessions are the replication unit
    (appropriate for one animal).

    Args:
        sessions: ``{label: (pose_zarr, spike_zarr[, spikes_dir])}`` mapping, an
            ``ExperimentSubject``, or session objects.
        node (str): Limb whose kinematic is decoded (e.g. ipsilateral hindpaw).
        target (str): Kinematic feature (``'speed'``, ``'velocity_y'``, ...).
        contra_node (str | None): If given, also decode it and the residual of
            ``node`` orthogonal to it (the independent-ipsi control).
        good_units: ``'auto'`` (curation 'good' per session), an explicit unit
            list, or ``None`` (all units in the store).
        select: Optional subset of sessions — a list/set of session labels to keep,
            or a ``callable(label) -> bool``. ``None`` (default) uses all sessions.
        bin_ms (int | None): Which binned store to use; defaults to the largest.
        n_splits, n_shuffle, smoothing_s, lag, alpha, all_events: decoder settings
            (held fixed across sessions for comparability).
        save_path (Path | str | None): If given, write ``decode_across_sessions.csv``.

    Returns:
        pd.DataFrame: rows of ``session, decode, target, n_units, cv_r2, cv_corr,
        shuffle_p`` (one per session × decode).
    """
    items = _iter_decode_sessions(sessions, bin_ms=bin_ms)
    if select is not None:
        if callable(select):
            items = [it for it in items if select(it[0])]
        else:
            keep = {str(s) for s in select}
            items = [it for it in items if str(it[0]) in keep]
    if not items:
        raise ValueError("No sessions to decode — none had matching binned stores, "
                         "or `select` filtered them all out.")

    rows = []
    for label, pose_p, spike_p, sdir in items:
        # resolve the decoding population
        units = None
        if good_units == "auto":
            g = _session_good_units(sdir)
            units = list(g) if g is not None else None
        elif isinstance(good_units, (list, tuple, set)):
            units = list(good_units)
        try:
            store_units = [int(u) for u in load_zarr(Path(spike_p), method='xarray').unit.values]
        except Exception:
            store_units = []
        units = store_units if units is None else [u for u in units if int(u) in set(store_units)]

        decodes = {"ipsi": {"node": node}}
        if contra_node:
            decodes = {"contra": {"node": contra_node},
                       "ipsi": {"node": node},
                       "ipsi_residual": {"node": node, "partial_out_node": contra_node}}

        for dname, kw in decodes.items():
            params = build_decoder_params(
                node=kw["node"], units=units, target=target, n_splits=n_splits,
                lag=lag, alpha=alpha, smoothing_s=smoothing_s, all_events=all_events,
                n_shuffle=n_shuffle, partial_out_node=kw.get("partial_out_node"))
            row = {"session": label, "decode": dname, "target": target,
                   "n_units": len(units)}
            try:
                _, _, out = create_glm_decoder(pose_p, spike_p, params)
                m = out["params"].get("metrics", {})
                row.update({"cv_r2": m.get("cv_r2"), "cv_corr": m.get("cv_corr"),
                            "shuffle_p": m.get("shuffle_p")})
            except Exception as e:
                row.update({"cv_r2": float('nan'), "cv_corr": float('nan'),
                            "shuffle_p": float('nan'), "error": str(e)})
            rows.append(row)

    summary = pd.DataFrame(rows)
    if save_path:
        save_path = Path(save_path)
        target_path = save_path if save_path.suffix else (save_path / 'decode_across_sessions.csv')
        target_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataframe(summary, target_path, storage_format='csv')
    return summary


def _wilcoxon_vs0(values):
    """One-sample Wilcoxon signed-rank of finite ``values`` against 0; NaN if too few."""
    from scipy import stats
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    d = v[v != 0]
    if d.size < 1:
        return float('nan')
    try:
        return float(stats.wilcoxon(d).pvalue)
    except Exception:
        return float('nan')


def decode_stats(summary):
    """Session-level stats for an across-session decode summary.

    Sessions are the replication unit (one animal). For each decode type it tests
    whether CV R² is above chance across sessions, and it compares decode types
    pairwise — so you can state, e.g., "contra > ipsi" and, crucially,
    "ipsi-residual > 0" (the independent-ipsilateral claim).

    Args:
        summary (pd.DataFrame | list): Output of
            :func:`decode_across_sessions` (or a list of them — concatenated).

    Returns:
        dict with:
          * ``per_decode`` (DataFrame): per decode — ``n_sessions``,
            ``median_cv_r2``, ``mean_cv_r2``, ``cv_r2_vs0_wilcoxon_p`` (population
            "above chance?"), and ``k_shuffle_sig`` / ``n_shuffle`` (per-session
            permutation hits).
          * ``pairwise`` (DataFrame): paired Wilcoxon between decode types
            (``contra`` vs ``ipsi``, ``ipsi`` vs ``ipsi_residual``, ...),
            Holm-corrected (``wilcoxon_p_holm``).
    """
    import itertools
    from scipy import stats

    if isinstance(summary, (list, tuple)):
        summary = pd.concat(list(summary), ignore_index=True)

    present = list(summary['decode'].unique())
    decodes = [d for d in ('contra', 'ipsi', 'ipsi_residual') if d in present]
    decodes += [d for d in present if d not in decodes]

    rows = []
    for d in decodes:
        sub = summary[summary['decode'] == d]
        r2 = np.asarray(sub['cv_r2'].values, dtype=float)
        r2 = r2[np.isfinite(r2)]
        if 'shuffle_p' in sub.columns:
            sp = np.asarray(sub['shuffle_p'].values, dtype=float)
            sp = sp[np.isfinite(sp)]
        else:
            sp = np.array([])
        rows.append({
            'decode': d, 'n_sessions': int(r2.size),
            'median_cv_r2': float(np.median(r2)) if r2.size else float('nan'),
            'mean_cv_r2': float(np.mean(r2)) if r2.size else float('nan'),
            'cv_r2_vs0_wilcoxon_p': _wilcoxon_vs0(r2),
            'k_shuffle_sig': int((sp < 0.05).sum()), 'n_shuffle': int(sp.size),
        })
    per_decode = pd.DataFrame(rows)

    piv = summary.pivot_table(index='session', columns='decode', values='cv_r2')
    pr = []
    for a, b in itertools.combinations(decodes, 2):
        if a in piv.columns and b in piv.columns:
            pair = piv[[a, b]].dropna()
            diff = pair[a].values - pair[b].values
            dnz = diff[diff != 0]
            try:
                p = float(stats.wilcoxon(dnz).pvalue) if dnz.size >= 1 else float('nan')
            except Exception:
                p = float('nan')
            pr.append({'decode_a': a, 'decode_b': b, 'n_sessions': int(len(pair)),
                       'median_diff': float(np.median(diff)) if len(pair) else float('nan'),
                       'wilcoxon_p': p})
    pairwise = pd.DataFrame(pr)
    if not pairwise.empty:
        from statsmodels.stats.multitest import multipletests
        pairwise['wilcoxon_p_holm'] = np.nan
        finite = pairwise['wilcoxon_p'].notna()
        if finite.any():
            pairwise.loc[finite, 'wilcoxon_p_holm'] = multipletests(
                pairwise.loc[finite, 'wilcoxon_p'].values, method='holm')[1]

    return {'per_decode': per_decode, 'pairwise': pairwise}