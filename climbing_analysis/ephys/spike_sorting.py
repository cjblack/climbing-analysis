from pathlib import Path
import yaml
from spikeinterface.sorters import run_sorter
from spikeinterface import create_sorting_analyzer
from spikeinterface.exporters import export_to_phy
from spikeinterface.extractors import read_phy
from climbing_analysis.ephys.utils import *


def sort_spikes(data_path: str, param_file:str) #rec_type:str = 'openephys', sorter='kilosort4', probe_manufacturer: str = 'cambridgeneurotech', probe_id: str = 'ASSY-236-H5', channel_map = 'h5_channel_map.npy'):
    """
    Sort spikes from data file - default is running kilosort4 on open ephys data recorded with H5 probe
    """
    # Load sorting params
    sorting_params = get_sorting_params(param_file)

    rec_type = sorting_params['rec_type']
    sorter = sorting_params['sorter']
    probe_id = sorting_params['probe_id']
    probe_manufacturer = sorting_params['probe_manufacturer']
    channel_map = sorting_params['channel_map']
    
    data_path = Path(data_path) # windows path
    output_folder = data_path / sorter # set output folder for kilosort
    recording_path = data_path / f'{sorter}/recording.dat'
    phy_folder = data_path / f'{sorter}/phy_output'

    recording = read_data(data_path=Path(data_path), rec_type=rec_type)
    probe = create_probe(probe_manufacturer, probe_id, channel_map) # creates probe from manufacturer, id, and channel map
    recording = recording.set_probe(probe, group_mode='by_shank') # sets probe

    # Run spikesorting
    sorting = run_sorter(sorter_name=sorter, recording=recording, folder=output_folder)

    # save recording as binary format to kilosort4 folder
    recording.save_to_folder(data_path=recording_path) # might not need to run this step...**
    analyzer = sorting_analyzer(sorting, recording, data_path) # create sorting analyzer
    export_to_phy(analyzer, output_folder=phy_folder) # export to phy for visualization
    return sorting, recording, probe, analyzer

def sorting_analyzer(sorting, recording, data_path):
    """
    Create sorting analyzer
    """
    folder = data_path / 'analyzer_folder'
    analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, format='binary_folder',return_in_uV=True,folder=folder)
    analyzer.compute(['random_spikes', 'waveforms', 'templates', 'noise_levels'])
    _ = analyzer.compute('spike_amplitudes')
    _ = analyzer.compute('principal_components', n_components=5, mode="by_channel_local")
    return analyzer

def load_phy_sorting(directory):
    """
    Load sorting data from phy2
    """
    sorting = read_phy(directory)
    return sorting

def plot_waveform(wfs, channel):
    """
    Plot spike waveforms
    """
    for x in range(300):
        plt.plot(wfs[x, :, channel], color='purple', alpha=0.5)
    plt.plot(np.mean(wfs[:,:,channel],axis=0),color='black', linewidth=2)
    plt.savefig('D:/ClimbingData/SOD1/WT/Example_unit72_waveform_average.pdf')
    plt.show()


def plot_session_psth(unit_ids, sorting, dflist, frame_captures, stances, node='r_forepaw', epoch_loc='start', xlim_=[-0.5,0.5], bin_size=0.02, smooth_sigma=1.0, save_fig=None):
    """
    Plot peristimulus time histogram from session
    """
    fig, ax =plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    spike_train_total = []
    for unit_id in unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
        spike_train_total.append(spike_train)
    spike_train_total = np.concatenate(spike_train_total)
    spike_train = spike_train_total/30000
    spikes_to_store = []
    spks_per_trial_total = []
    all_counts = []
    bins = np.arange(xlim_[0], xlim_[1] + bin_size, bin_size)
    iii = 0

    # Sort sessions in numerical order
    trial_ids = []
    for df in dflist:
        tid = int(df.attrs['Trial'].split('T')[-1])
        trial_ids.append(tid)
    trial_ids_sort = np.argsort(trial_ids)

    # Get spike rasters for each movement across climbing trials for single session
    for ii in range(len(frame_captures)):
        bt = frame_captures[ii]/30000
        bout_start_id = len(bt) - (dflist[trial_ids_sort[ii]].__len__())
        times_ = np.array(stances[trial_ids_sort[ii]][node][epoch_loc])
        aligned_spikes = spike_train - bt[bout_start_id]
        spks_per_trial = []
        for i, tstart in enumerate(times_):
            tstart = tstart/200.0
            spikes_in_window = aligned_spikes[(aligned_spikes>(tstart-0.5)) & (aligned_spikes <=(tstart+0.5))]
            spikes_to_store.append(spikes_in_window-tstart)
            spks_per_trial.append([i,spikes_in_window-tstart])

            counts, _ = np.histogram(spikes_in_window-tstart, bins=np.arange(xlim_[0],xlim_[1]+bin_size,bin_size))
            all_counts.append(counts)
        spks_per_trial_total.append(spks_per_trial)

    sorted_spikes = sorted(spikes_to_store,key=len, reverse=True)
    for iii, x in enumerate(sorted_spikes):
        ax[0].vlines(x, iii + 0, iii + 1, color='black', lw=1)
    ax[0].axvline(0.0, linestyle='--', color='red', linewidth=0.75,alpha=0.5)
    ax[0].set_xlim(xlim_)
    ax[0].set_ylabel('Trial')
    if len(all_counts) > 0:
        all_counts = np.array(all_counts)
        firing_rate = (all_counts / bin_size)
        mean_rate = np.mean(firing_rate,axis=0)
        std_rate = np.std(firing_rate,axis=0)
        smoothed_rate = gaussian_filter1d(mean_rate, sigma=smooth_sigma)
        tbins = bins[:-1] + bin_size/2
        ax[1].plot(tbins,smoothed_rate,color='black')
        ax[1].axvline(0, linestyle='--',color='red', linewidth=0.75, alpha=0.5)
        ax[1].set_ylabel('Firing rate (spikes/s)')
        ax[1].set_xlabel('Time (s)')

    plt.title(f'{node} movement {epoch_loc}')
    if save_fig:
        plt.savefig(save_fig+f'/{node}_movement-{epoch_loc}_unit_id{unit_ids[0]}_spikeraster.png')
        plt.savefig(save_fig+f'/{node}_movement-{epoch_loc}_unit_id{unit_ids[0]}_spikeraster.pdf')

    plt.show()
    return spikes_to_store, spks_per_trial_total