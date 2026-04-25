import matplotlib.pyplot as plt
import numpy as np

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