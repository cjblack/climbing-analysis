
from pathlib import Path

import pandas as pd
import numpy as np
import zarr

from neurokinematics.io import load_csv, load_zarr

def get_movement_aligned_erps(alignment: pd.DataFrame | Path | str, lfp_data: dict | Path | str, save_path: Path | str, storage_format: str = 'pickle', channel_select: np.ndarray | None = None):

    
    # load alignment
    if isinstance(alignment, (Path, str)):
        alignment_path = Path(alignment)
        if alignment_path.exists():
            alignment = load_csv(alignment_path, pkg_format='pandas')
        else:
            raise FileNotFoundError(f'Alignment file not found {alignment}.')
        
    # load lfp
    if isinstance(lfp_data, (Path, str)):
        lfp_path = Path(lfp_data)
        if lfp_path.exists():
            lfp, meta = load_zarr(lfp_path, dataset='processed')
            timestamps = load_zarr(lfp_path, dataset='time')[0][:] # load into memory
            channels = load_zarr(lfp_path, dataset='channel')[0]
    else:
        lfp = lfp_data['lfp']
        meta = lfp_data['meta']
        timestamps = lfp_data['timestamps']
        channels = lfp_data['channels']

    # select channels
    if channel_select is None:
        channel_select = channels


    if isinstance(channel_select, np.ndarray):
        if (len(channel_select) > lfp.shape[1]):
            print("Selected more channels than available. Defaulting to all channels")
            channel_select = channels
        elif not (np.all(np.isin(channel_select, channels))):
            print("Channel select ids out of range. Defaulting to all channels")
            channel_select = channels

    channel_select = np.asarray(channel_select)
    channel_idxs = np.where(np.isin(channels, channel_select))[0]

    # set sampling rate to processed fs
    fs = meta['fs_processed']

    # 500ms pre and 500ms pose including event sample
    pre_s = 0.5
    post_s = 0.5
    pre_samples = int(pre_s*fs)
    post_samples = int(post_s*fs)
    n_timepoints = pre_samples + post_samples + 1

    time_axis = np.arange(-pre_samples, post_samples+1) / fs
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(save_path.as_posix()), mode="w") # check if zarr has issue with type Path

    # metadata
    root.attrs["fs"] = fs
    root.attrs["pre_s"] = pre_s
    root.attrs["post_s"] = post_s
    root.attrs["n_timepoints"] = n_timepoints

    root.create_dataset("channels", data=channel_select)
    root.create_dataset("time", data=time_axis)

    for nd in alignment["node"].unique():
        node_df = alignment.query("node==@nd") # this may be an issue if nd names are awkward (i.e. various characters)
        node_group = root.require_group(str(nd))

        for me in node_df["movement_event"].unique():
            event_df = node_df.query("movement_event==@me").copy()
            event_df = event_df.reset_index(drop=True)
            
            event_group = node_group.require_group(str(me))

            n_occurrences = len(event_df)

            epochs = event_group.create_dataset(
                "movement_epochs",
                shape=(n_occurrences, len(channel_idxs), n_timepoints),
                chunks=(min(64, n_occurrences), len(channel_idxs), n_timepoints),
                dtype=lfp.dtype,
                fill_value=np.nan
            )

            valid = np.zeros(n_occurrences, dtype=bool)

            for i, row in event_df.iterrows():
                #center_sample = int(row["event_times_samples"])
                center_ts = row["event_times_ts"]
            
                sample_index = np.argmin(np.abs(timestamps-center_ts)) # this may need to be optimised

                start = sample_index - pre_samples
                stop = sample_index + post_samples + 1
                if start < 0 or stop > lfp.shape[0]:
                    continue

                epochs[i, :, :] = lfp[start:stop, channel_idxs].T
                valid[i] = True

            event_group.create_dataset("valid", data=valid)
            event_group.create_dataset("video_id", data=event_df["trial"].to_numpy())
            event_group.create_dataset("frame_ids", data=event_df["frame_ids"].to_numpy())
            event_group.create_dataset("event_time_ts", data=event_df["event_times_ts"].to_numpy())
            event_group.create_dataset("event_time_samples", data=event_df["event_times_samples"].to_numpy())
    return root