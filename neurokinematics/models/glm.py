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
from sklearn.model_selection import KFold  # optional: requires scikit-learn; if unavailable, use custom CV
from scipy import __version__ as scipy_version

from sklearn.preprocessing import StandardScaler
from sklearn import __version__ as sk_version

import statsmodels.api as sm
from statsmodels import __version__ as sm_version

import xarray as xr

from neurokinematics.io import load_zarr, save_model, save_yaml, save_dataset, save_dataframe
from neurokinematics import __version__ as nk_version


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
        Defaults to None.

    Raises:
        ValueError: Raises when pose_ds is str or Path, and the file format for pose_ds is invalid
        ValueError: Raises when spike_ds is str or Path, and the file format for spike_ds is invalid

    Returns:
        _type_: _description_
    """


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

    if params is None:
        params = {}

    params['input_data'] = {'pose_dataset': pose_ds_str, 'spike_dataset': spike_ds_str}
    node = params.get("pose", {}).get("node", pose_ds.node.values[0]) #glm_params['node']
    family = params.get("family", 'Poisson')
    glm_type = params.get("type", 'encoder')  #glm_params['type']
    pose_feature = params.get("pose", {}).get('features', ['position_y']) #glm_params['features']['pose']
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

    predicted = results.predict(X_model)

    params['packages'] = {'statsmodels': sm_version, 'scipy': scipy_version, 'sklearn': sk_version, 'neurokinematics': nk_version}
    params['metrics'] = {'aic': float(results.aic), 'log_likelihood': float(results.llf)}

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

    if params is None:
        params = {}

    params['input_data'] = {'pose_dataset': pose_ds_str, 'spike_dataset': spike_ds_str}
    node = params.get("pose", {}).get("node", pose_ds.node.values[0]) #glm_params['node']
    family = params.get("family", 'Gaussian') # set gaussian by default for decoding
    glm_type = params.get("type", 'encoder')  #glm_params['type']
    pose_feature = params.get("pose", {}).get('features', ['position_y'])[0] #glm_params['features']['pose']
    pose_feature, pose_coord = pose_feature.split('_')
    spike_feature = params.get("spikes", {}).get('features', 'spike_counts') #glm_params['features']['spikes']
    spike_feature = spike_feature[0] # this will be a list - but should only contain one entry
    units = params.get("spikes", {}).get('unit', 0) # this will be a list - but should only contain one entry
    time_bins = spike_ds.time_bin.values

    attrs = {
        "model_type": glm_type,
        "unit": units,
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
    for i, uid in enumerate(units):
        feat_ = spike_sub[spike_feature].isel(unit=uid)
        features.append(f"unit_{uid}")
        predictors[f"unit_{uid}"] = feat_
        if i == 0:
            n_events, n_bins = feat_.shape
    X = pd.DataFrame({name: predictors[name].values.reshape(-1) for name in features})

    #spikes = spike_sub[spike_feature].sel(unit=unit)#.isel(time_bin=slice(1, None))
    #n_events, n_bins = spikes.shape

    event_idx = np.repeat(np.arange(n_events), n_bins)
    time_idx = np.tile(np.arange(n_bins), n_events)

    valid = (
        pose_sub.valid.fillna(False).astype(bool) &#.isel(time_bin = slice(1, None)) &
        spike_sub.valid.fillna(False).astype(bool)#.isel(time_bin = slice(1, None))
    )

    sy = pose_sub[pose_feature].sel(node=node).sel(coord=pose_coord)#spikes.values.reshape(-1)

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

    predicted = results.predict(X_model)

    params['packages'] = {'statsmodels': sm_version, 'scipy': scipy_version, 'sklearn': sk_version, 'neurokinematics': nk_version}
    params['metrics'] = {'aic': float(results.aic), 'log_likelihood': float(results.llf)}

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
        save_path = save_path / 'glm' / glm_type / f'population_to_{node}_{pose_feature}_{created_on}'
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
    plot_on = params['plot']
    glm_type = params['type']
    mode = params.get('comparison', {}).get('mode', 'full')

    if glm_type == 'encoder':
        node = params.get('pose', {}).get('node', x_ds.node.values[0])
        unit = params.get('spikes', {}).get('unit', 0)
        features = params.get('pose', {}).get('features', ['position_x'])
        glm_save_directory = f'comparison_{mode}_{node}_unit_{unit}_{created_on}'

    #if glm_type == 'decoder':
         
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
            save_glm_results(model, results, outputs, params, save_path / model_name)
            
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