from pathlib import Path
from copy import deepcopy

import pymc as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from neurokinematics.io import load_parquet

def _build_prior(name:str, config: dict):

    dist_name = config.pop('dist')
    dist = getattr(pm, dist_name)

    return dist(name, **config)

def fit_hierarchical_linear(df: str | Path | pd.DataFrame, params: dict | None = None, save_path: str | Path | None = None):
    """_summary_

    Args:
        df (str | Path | pd.DataFrame): format
            {
            'node': 'node',
            'feature': 'feature',
            'samples': 2000,
            'tune': 1000,
            'seed': 42, # int for seed
            'chains': 4,
            'predictor': ['session_number'],
            'likelihood': 'Normal',
            'priors': {
                'group_baseline': {'dist': 'Normal', 'mu': 90, 'sigma': 10},
                'group_slope': {'dist': 'Normal', 'mu': 0, 'sigma': 5},
                'sigma_baseline': {'dist': 'HalfNormal', 'sigma': 10},
                'sigma_slope': {'dist': 'HalfNormal', 'sigma': 2},
                'sigma_obs': {'dist': 'HalfNormal', 'sigma': 10},
                'subject_baseline': {'dist': 'Normal'}, # requires mu and sigma from group level distributions
                'subject_slope': {'dist': 'Normal'} # requires mu and sigma from group leve distributions
                }
            }
        params (dict): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(df, (str, Path)):
        df = load_parquet(df, method="pandas")

    default_params = {
            'node': df['node'].unique()[0],
            'feature': 'v_mag_max',
            'samples': 2000,
            'tune': 1000,
            'seed': 42,
            'chains': 4,
            'predictor': ['session_number'],
            'likelihood': 'Normal',
            'priors': {
                'group_baseline': {'dist': 'Normal', 'mu': 90, 'sigma': 10},
                'group_slope': {'dist': 'Normal', 'mu': 0, 'sigma': 5},
                'sigma_baseline': {'dist': 'HalfNormal', 'sigma': 10},
                'sigma_slope': {'dist': 'HalfNormal', 'sigma': 2},
                'sigma_obs': {'dist': 'HalfNormal', 'sigma': 10},
                'subject_baseline': {'dist': 'Normal'}, # requires mu and sigma from group level distributions
                'subject_slope': {'dist': 'Normal'} # requires mu and sigma from group leve distributions
                }
            }
    
    if params is None:
        params = default_params

    node = params['node']
    samples = params['samples']
    tune = params['tune']
    seed = params['seed']
    feature = params['feature']
    predictors = params['predictor']
    priors = params['priors']
    likelihood = params['likelihood']

    if isinstance(predictors, str):
        predictors = [predictors]

    # check predictors don't overlap with feature
    overlapping = set(predictors) & {feature}
    if overlapping:
        raise ValueError(f"Predictor(s) {overlapping} overlap with the outcome feature '{feature}'. "
                         f"Predictors and outcome must be different columns."
                         )
    
    # check predictors exist
    missing = set(predictors) - set(df.columns)
    if missing:
        raise ValueError(
            f"Predictor(s) {missing} not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # check feature exists
    if feature not in df.columns:
        raise ValueError(
            f"Feature '{feature}' is not found in DataFrame. "
            f"Available features: {list(df.columns)}"
        )


    df_sub = df.query("node==@node").copy()
    df_sub = df_sub.sort_values(['id', 'date'])

    # index data
    subject_idx = pd.Categorical(df_sub['id']).codes
    n_subjects = len(np.unique(subject_idx))
    n_predictors = len(predictors)

    #session_number = df_sub['session_number'].values # 
    X = df_sub[predictors].values
    data = df_sub[feature].values
    if n_predictors > 1:
        X = (X-X.mean(axis=0)) / X.std(axis=0) # standardize if there are more than one predictor
        #data = df_sub[feature].values
        y_mean = data.mean()
        y_std = data.std()
        data = (data - y_mean) / y_std


    # taking the form: Y = B0 + B1X + error
    with pm.Model() as hierarchical:
        # GROUP LEVEL
        # priors
        group_baseline = _build_prior('group_baseline', deepcopy(priors['group_baseline']))
        group_slope = _build_prior('group_slope', deepcopy(priors['group_slope']))
        
        # subject variety
        sigma_baseline = _build_prior('sigma_baseline', deepcopy(priors['sigma_baseline']))
        sigma_slope = _build_prior('sigma_slope', {**deepcopy(priors['sigma_slope']), 'shape': n_predictors})
        
        # SUBJECT LEVEL
        subject_baseline = getattr(pm, deepcopy(priors['subject_baseline']['dist']))(
            'subject_baseline', 
            mu = group_baseline, 
            sigma = sigma_baseline, 
            shape = n_subjects
            )
        subject_slope = getattr(pm, deepcopy(priors['subject_slope']['dist']))(
            'subject_slope', 
            mu = group_slope,
            sigma = sigma_slope,
            shape = (n_subjects, n_predictors)
            )

        # Noise
        sigma_obs = _build_prior('sigma_obs', priors['sigma_obs'])

        # Linear model
        mu = subject_baseline[subject_idx] + (subject_slope[subject_idx] * X).sum(axis=1)#session_number

        # likelihood
        data_obs = getattr(pm, likelihood)(
                feature,
                mu = mu,
                sigma = sigma_obs,
                observed = data
        )

        
        target_accept = params.get('target_accept', 0.9)
        chains        = params.get('chains', 4)

        # Sample normally — chains run in parallel where the platform allows.
        # progressbar disabled: PyMC's rich Live display garbles the GUI log.
        trace = pm.sample(
            samples,
            tune=tune,
            chains=chains,
            return_inferencedata=True,
            random_seed=seed,
            target_accept=target_accept,
            progressbar=False,
        )

    # save trace as pickle (model excluded — not picklable due to compiled pytensor functions)
    if save_path is not None:
        import pickle
        from datetime import datetime
        save_path    = Path(save_path)
        bayesian_dir = save_path / 'bayesian'
        bayesian_dir.mkdir(parents=True, exist_ok=True)
        date_str   = datetime.now().strftime('%Y%m%d_%H%M%S')
        trace_path = bayesian_dir / f'hierarchical_linear_{date_str}.pkl'
        with open(trace_path, 'wb') as f:
            pickle.dump({'trace': trace, 'params': params}, f)
        print(f"Trace saved to {trace_path}")

    return hierarchical, trace
