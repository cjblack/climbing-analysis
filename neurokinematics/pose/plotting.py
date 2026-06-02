from pathlib import Path
import xarray as xr

import matplotlib.pyplot as plt
import numpy as np

from neurokinematics.io import load_zarr
from neurokinematics.pose.utils import pixels_to_cm

def plot_phase_offset_pairs(poff: dict, bin_size: int = 25, phase_mode='default'):
    no_pairs = len(poff.keys())
    if no_pairs <= 2:
        fig, ax  = plt.subplots(ncols=no_pairs)
    else:
        fig, ax = plt.subplots(nrows=no_pairs)
    
    
    if phase_mode == 'default':
        de_factor = 1
    elif phase_mode == 'radians':
        de_factor = 2.*np.pi
    elif phase_mode == 'degrees':
        de_factor = 360.

    bins = np.linspace(0,1*de_factor,bin_size)
    
    for i, (npair, vals) in enumerate(poff.items()):
        phase_ = np.concat(vals['values'])*de_factor
        ax[i].hist(phase_, bins, density=True)
        ax[i].set_xlabel(f'Phase offset ({phase_mode})')
        ax[i].set_ylabel('Density')
        ax[i].set_title(f'{npair[0]} - {npair[1]}')
        ax[i].set_xlim([-de_factor/10,de_factor*1.1])
    
    plt.tight_layout()
    plt.show()

def plot_pose_trajectory(movement_ds: str | Path | xr.Dataset, params: dict, save_path: str | Path | None = None):

    
    if isinstance(movement_ds, (str, Path)):
        movement_ds = Path(movement_ds)
        if movement_ds.suffix == '.zarr':
            movement_ds = load_zarr(movement_ds, method='xarray')
        else:
            raise ValueError('Accepted file formats are: ".zarr".')

    
    node = params['node']
    feature = params['feature']
    fig_formats = params.get('formats', ['.png', '.pdf'])
    mask = (movement_ds.reference_node == node).compute()
    movement_sub = movement_ds.where(mask, drop=True)
    no_events = movement_sub.event.values.shape[0]
    
    x = movement_sub[feature].sel(coord='x').values * pixels_to_cm()
    y = movement_sub[feature].sel(coord='y').values * pixels_to_cm()
    
    for i in range(no_events):
        x_ = x[i]
        y_ = y[i]
        x_start = x_[0]
        y_start = y_[0]
        x_ = x_-x_start
        y_ = y_-y_start
        plt.plot(x_, y_, color='black', alpha=0.2)
    
    if feature == 'position':
        units = 'cm'
    if feature == 'velocity':
        units = 'cm/s'
    if feature == 'acceleration':
        units = 'cm/s2'
    
    plt.xlabel(f'{feature} ({units})')
    plt.ylabel(f'{feature} ({units})')

    plt.title(f'{node} {feature}')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        fig_name = f'trajectory_{node}_{feature}'
        for format in fig_formats:
            plt.savefig(save_path / fig_name+format)

        
    plt.show()