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
            'predictor': 'session_number',
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
            'feature': 'v_magnitude',
            'samples': 2000,
            'tune': 1000,
            'seed': 42,
            'chains': 4,
            'predictor': 'session_number',
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
    feature = params['feature']
    samples = params['samples']
    tune = params['tune']
    seed = params['seed']
    predictor = params['predictor']
    priors = params['priors']
    likelihood = params['likelihood']


    df_sub = df.query("node==@node").copy()
    df_sub = df_sub.sort_values(['id', 'date'])

    # index data
    subject_idx = pd.Categorical(df_sub['id']).codes

    #session_number = df_sub['session_number'].values # 
    x = df_sub[predictor].values
    data = df_sub[feature].values

    n_subjects = len(np.unique(subject_idx))

    # taking the form: Y = B0 + B1X + error
    with pm.Model() as hierarchical:
        # GROUP LEVEL
        # priors
        group_baseline = _build_prior('group_baseline', deepcopy(priors['group_baseline']))
        group_slope = _build_prior('group_slope', deepcopy(priors['group_slope']))
        
        # subject variety
        sigma_baseline = _build_prior('sigma_baseline', deepcopy(priors['sigma_baseline']))
        sigma_slope = _build_prior('sigma_slope', deepcopy(priors['sigma_slope']))
        
        # SUBJECT LEVEL
        subject_baseline = getattr(pm, priors['subject_baseline']['dist'])(
            'subject_baseline', 
            mu = group_baseline, 
            sigma = sigma_baseline, 
            shape = n_subjects
            )
        subject_slope = getattr(pm, priors['subject_slope']['dist'])(
            'subject_slope', 
            mu = group_slope,
            sigma = sigma_slope,
            shape = n_subjects
            )

        # Noise
        sigma_obs = _build_prior('sigma_obs', priors['sigma_obs'])

        # Linear model
        mu = subject_baseline[subject_idx] + subject_slope[subject_idx] * x#session_number

        # likelihood
        data_obs = getattr(pm, likelihood)(
                feature,
                mu = mu,
                sigma = sigma_obs,
                observed = data
        )

        
        trace = pm.sample(samples, tune = tune, return_inference_data = True, random_seed = seed)

    return hierarchical, trace
