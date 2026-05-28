import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
import pymc as pm

from scipy.stats import uniform, norm

import matplotlib.pyplot as plt
import seaborn as sns

from neurokinematics.pose.utils import pixels_to_cm
from neurokinematics.io import load_zarr

def create_grid(x: float, y: float, size: int):
    grid = np.linspace(x, y, size)
    return grid

def get_prior(params: dict, grid: np.ndarray, size: int = 1000):
    
    prior = norm.pdf(grid, params['mean'], params['std'])
    return prior

def get_likelihood(data, grid):
    likelihood = np.zeros(grid.shape)
    for ig, ug in enumerate(grid):
        likelihood[ig] = norm.logpdf(data,ug, data.std()).sum()
    likelihood -= likelihood.max()
    likelihood = np.exp(likelihood)
    return likelihood

def get_data(ds: xr.Dataset):
    date = ds.date.values[0]
    id = ds.id.values[0]
    node = 'r_forepaw'
    mask = (ds.reference_node == node).compute()
    ds_sub = ds.where(mask, drop=True)
    vy = ds_sub.velocity.sel(coord='y', node = node)
    data = np.nanmax(vy, axis=1) * pixels_to_cm()

    return data, id, date

def compute_posterior(likelihood, prior, grid):
    posterior = likelihood * prior
    dx = grid[1]-grid[0]
    posterior /= (posterior.sum())# * dx)

    return posterior


def bayesian_grid_search(x, y, size, params):
    ds = load_zarr(params['dataset'], method='xarray')
    data, id, date = get_data(ds)
    date = str(np.datetime_as_string(date, unit='D'))
    grid = create_grid(x, y, size)
    prior = get_prior(params, grid, size)
    likelihood = get_likelihood(data, grid)
    posterior = compute_posterior(likelihood, prior, grid)

    df_data = pd.DataFrame({
        'velocity': data,
        'update': 0,
        'id': id,
        'date': date
    })

    dfs = []
    for dist_type, values in zip(['prior', 'likelihood', 'posterior'], [prior, likelihood, posterior]):
        dfs.append(pd.DataFrame({
            'grid': grid,
            'density': values,
            'type': dist_type,
            'update': 0,
            'date': date,
            'id': id
        }))



    df_bayes = pd.concat(dfs, ignore_index = True)

    return df_bayes, df_data


def update_posterior(df_bayes, df_data, ds):
    grid = df_bayes['grid'].unique()
    ds = load_zarr(ds, method='xarray')
    data, id, date = get_data(ds)
    date = str(np.datetime_as_string(date, unit='D'))
    
    num_updates = len(df_bayes['update'].unique()) - 1 # minus one to account for 0th indexing
    

    updated_prior = df_bayes.query('update==@num_updates & type=="posterior"')['density'].values
    updated_likelihood = get_likelihood(data, grid)
    updated_posterior = compute_posterior(updated_likelihood, updated_prior, grid)
    
    df_data_new = pd.DataFrame({
        'velocity': data,
        'update': num_updates+1,
        'id': id,
        'date': date
    })

    update = []
    for dist_type, values in zip(['prior', 'likelihood', 'posterior'], [updated_prior, updated_likelihood, updated_posterior]):
        update.append(pd.DataFrame({
            'grid': grid,
            'density': values,
            'type': dist_type,
            'update': num_updates+1,
            'date': date,
            'id': id
        }))

    update = pd.concat(update, ignore_index = True)
    df_bayes = pd.concat([df_bayes, update], ignore_index = True)
    df_data = pd.concat([df_data, df_data_new], ignore_index = True)
    return df_bayes, df_data

def plot_distributions(df_bayes, df_data, updates: list | None = None, axes = None):
    
    sns.set_style('darkgrid')    
    
    posteriors = df_bayes[df_bayes['type']=='posterior']
    likelihoods = df_bayes[df_bayes['type'] == 'likelihood']
    if isinstance(updates, list):
        if set(updates).issubset(posteriors['update'].unique()):
            posteriors = posteriors.query("update==@updates")
            likelihoods = likelihoods.query("update==@updates")
            df_data = df_data.query("update==@updates")
        else:
            raise ValueError(f"updates must match valid update indices in df_bates, such as: {df_bayes['update'].unique()}")
    
    if axes is None:
        fig, axes = plt.subplots(nrows=3)

    sns.lineplot(data=posteriors, x='grid', y='density', hue='update', palette='crest', ax=axes[0])
    sns.lineplot(data=likelihoods,x='grid', y='density', hue='update', palette='crest', ax=axes[1])
    sns.kdeplot(data=df_data, x='velocity', hue='update', palette='crest', ax=axes[2], fill=True)
    axes[0].set_title('Posterior',fontsize=14)
    axes[1].set_title('Likelihood',fontsize=14)
    axes[2].set_title('Data', fontsize=14)
    xlims = axes[0].get_xlim()
    axes[1].set_xlim(xlims)
    axes[2].set_xlim(xlims)

    #plt.tight_layout()
    #plt.show()

def plot_credible_intervals(samples: dict, hdi_prob=0.95, axes = None):
    if axes is None:
        fig, axes = plt.subplots()
    az.plot_forest(samples, hdi_prob=hdi_prob, figsize=(6,8), textsize=12, ax=axes)
    plt.title('Credible intervals', fontsize=14)
    plt.xlabel('velocity (cm/s)', fontsize=12)
    plt.tight_layout()
    plt.show()

def create_uneven_axes(rows=3, cols=2):
    fig = plt.figure(figsize=(10,8))
    gs = fig.add_gridspec(rows,cols)
    ax_left = []
    ax_right = []
    for i in range(rows):
        ax_left.append(fig.add_subplot(gs[i,0]))
    ax_right.append(fig.add_subplot(gs[:,1]))

    return ax_right, ax_left
    
def compute_hdi(df_bayes, distribution: str, updates: list, size=10000, hdi_prob: float = 0.95):
    
    grid = df_bayes['grid'].unique()
    rows = []
    samples_dict = {}
    if distribution == 'diff_posterior':
        if len(updates) != 2:
            raise ValueError(f'updates list must comtain 2 values, it currently contains {len(updates)} values.')
        update_a = updates[0]
        update_b = updates[1]
        dist_a = df_bayes.query("type=='posterior' & update==@update_a")['density'].values
        dist_b = df_bayes.query("type=='posterior' & update==@update_b")['density'].values
        dist = dist_b - dist_a
        samples = np.random.choice(grid, size=size, p = dist)
        samples_dict['update'] = samples
        hdi = az.hdi(samples, hdi_prob = hdi_prob)
        rows.append({
            'update': 'diff',
            'mean': samples.mean(),
            'lower': hdi[0],
            'upper': hdi[1],
            'comparison': f"update {update_b} - update {update_a}"
        })
        df_hdi = pd.DataFrame(rows)
    else:
        for update in updates:
            dist = df_bayes.query("type==@distribution & update==@update")['density'].values
            samples = np.random.choice(grid, size=size, p = dist)
            samples_dict[f'update_{update}'] = samples
            hdi = az.hdi(samples, hdi_prob = hdi_prob)
            rows.append(
                {
                    'update': update,
                    'mean': samples.mean(),
                    'lower': hdi[0],
                    'upper': hdi[1]
                }
            )
        df_hdi = pd.DataFrame(rows)
    
    return df_hdi, samples_dict