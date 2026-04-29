"""Plot preprocessed LFP data.

Contains functions that primarily perform plotting, with some minimal signal analysis for visualisation.
"""

from pathlib import Path
import zarr

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def plot_movement_erps_probe(epoch_path: Path | str, channels: list, movement_plot_params: dict, save_path: Path | str | None = None):
    """Plot average lfp evoked responses based on movement events. Mainly for visualisation.

    Args:
        epoch_path (Path | str): Path to epoched lfp zarr store.
        channels (list): List of channels to plot.
        movement_plot_params (dict): Dictionary determining the plot, includes:
            {
                'node': 'node', # required - sets node for alignment
                'movement_event': 'movement_event', # required - sets movement event to align to
                'cmap': 'colormap', # optional - sets color of averages for plotting, defaults to 'black'
                'baseline_correct': False, # optional - sets whether evoked responses are baseline corrected (basline subtraction between 0.5-0.1 pre event) 
                'smooth': False, # optional - sets whether to smooth the averaged erp, if True, will use gaussian filtering with a sigma of 5 samples.
                'vertical_shift': 10 # optional - sets y-scale shift factor in the event of plotting multiple channels so traces don't overlap, optional - defaults to 10
            }
        save_path (Path | str | None, optional): Path to folder for saving plots to. If set, saves as `.png` to output folder. Defaults to None.

    Example:
        >>> plot_movement_erps_probe(
        ...     epoch_path = "path/to/zarr/store",
        ...     channels = [0,1,2,3,4,5],
        ...     movement_plot_params = {
        ...         'node': 'node',
        ...         'movement_event': 'start',
        ...         'baseline_correct': True,
        ...         'smooth': True,
        ...         }
        ...     )
    """

    node = movement_plot_params['node']
    movement_event = movement_plot_params['movement_event']

    if 'cmap' in movement_plot_params:
        if movement_plot_params['cmap'] == 'default':
            cmap = lambda i: 'black'
        else:
            cmap = plt.get_cmap(movement_plot_params['cmap'])
    else:
        cmap = lambda i: 'black'
    
    if 'baseline_correct' in movement_plot_params:
        bl_corr = movement_plot_params['baseline_correct']
    else:
        bl_corr = False
    
    if 'smooth' in movement_plot_params:
        erp_smooth = movement_plot_params['smooth']
    else:
        erp_smooth = False
    
    if 'xlims' in movement_plot_params:
        xlims = movement_plot_params['xlims']
    else:
        xlims = (-0.5,0.5)
    
    if 'vertical_shift' in movement_plot_params:
        vertical_shift = movement_plot_params['vertical_shift']
    else:
        vertical_shift = 10

    # load epoched lfp data
    epoch_path = Path(epoch_path)
    root = zarr.open_group(str(epoch_path), mode='r')
    fs = root.attrs['fs'] # 1000. # hard coded for testing
    
    # get specified epoch data
    group = root[node][movement_event]
    epochs = group["movement_epochs"][:]
    valid = group["valid"][:]
    time = root["time"][:]
    channel_ids = root["channels"][:]

    epochs = epochs[valid]
    if not isinstance(channels, list):
        channels = list(channels)
    
    if bl_corr:
        baseline_correct = time < -0.1
        baseline = np.nanmean(epochs[:, :, baseline_correct], axis=2, keepdims=True)
        epochs = epochs - baseline

    erp = np.nanmean(epochs, axis=0)

    if erp_smooth:
        sigma_samples = 5
        sigma_samples = sigma_ms / 1000 * fs
        erp = gaussian_filter1d(
            erp,
            sigma = sigma_samples,
            axis=1
        )
    
    fig, ax = plt.subplots(figsize=(4,10))
    
    for i, chan in enumerate(channels):
        ax.plot(time, erp[chan,:]+(vertical_shift*i), color = cmap(i/len(channels)), linewidth=2, label=f'chan {chan}')
    plt.xlim(xlims[0],xlims[1])
    plt.axvline(0.0, color = 'red', linestyle='--', linewidth=3)
    plt.ylabel('Voltage (mV)')
    plt.xlabel('Time (s)')
    plt.suptitle(f'{node} {movement_event} movement: lfp evoked response')
    plt.legend(loc = 'lower right', fontsize='xx-small')
    plt.tight_layout()

    if save_path:
        plots_dir = Path(save_path) / 'lfp_plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / f'{node}_{movement_event}_average_erp_across_chans.png'
        plt.savefig(plot_path.as_posix()) # save figure to analyzer path

    plt.show()