from pathlib import Path

import pymc as pm
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from neurokinematics.io import load_parquet

def fit_linear(df: str | Path | pd.DataFrame, params: dict):
    
    if isinstance(df, (str, Path)):
        df = load_parquet(df, method="pandas")
    node = params['node']
    sort_values = params['sort_values']
    group = params['group']
    subject = params['subject']
    error = params['error']
    noise = params['noise']
    feature = params['feature']
    tparams = params['trace']
    df_sub = df.query("node==@node").copy()
    df_sub = df_sub.sort_values(['id', 'date'])

    # index data
    subject_idx = pd.Categorical(df_sub['id']).codes
    session_number = df_sub['session_number'].values
    data = df_sub[feature].values

    n_subjects = len(np.unique(subject_idx))

    # taking the form: Y = B0 + B1X + error
    with pm.Model() as hierarchical:
        # GROUP LEVEL

        # priors
        group_baseline = pm.Normal(
            'group_baseline',
            mu = group['baseline']['mu'],
            sigma = group['baseline']['sigma'],
            )
        
        group_slope = pm.Normal(
            mu = group['slope']['mu'],
            sigma = group['slope']['sigma']
        )


        # subject variety
        sigma_baseline = pm.HalfNormal('sigma_baseline', sigma = error['baseline']['sigma'])
        sigma_slope = pm.HalfNormal('sigma_slope', sigma = error['baseline']['sigma'])


        # SUBJECT LEVEL
        # priors pulled from group distribution
        subject_baseline = pm.Normal('subject_baseline',
                                     mu = group_baseline,
                                     sigma = sigma_baseline,
                                     shape = n_subjects
                                     )
        
        subject_slope = pm.Normal('subject_slope',
                                  mu = group_slope,
                                  sigma = sigma_slope,
                                  shape = n_subjects
                                  )
        
        # Noise
        sigma_obs = pm.HalfNormal('sigma_obs', sigma = noise['sigma'])

        # Linear model
        mu = subject_baseline[subject_idx] + subject_slope[subject_idx] * session_number

        # likelihood
        data_obs = pm.Normal(feature,
                                 mu = mu,
                                 sigma = sigma_obs,
                                 observed = data
                                 )
        
        trace = pm.sample(tparams['samples'], tune = tparams['tune'], return_inference_data = True)

    return hierarchical, trace
