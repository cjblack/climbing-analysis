import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt, iirnotch, decimate

def filter_lfp(lfp, fs, notch_freq=50.0, band=(0.1, 100.0), quality=30.0):
    """
    Apply a 50 Hz notch and 0.1–100 Hz bandpass filter to an LFP signal.

    Parameters
    ----------
    lfp : np.ndarray
        1D array containing the LFP signal.
    fs : float
        Sampling frequency in Hz.
    notch_freq : float, optional
        Frequency to notch out (default = 50 Hz).
    band : tuple, optional
        Bandpass frequency range (low, high) in Hz (default = (0.1, 100)).
    quality : float, optional
        Quality factor for the notch filter (default = 30).

    Returns
    -------
    filtered_lfp : np.ndarray
        The filtered LFP signal.
    """
    
    # --- Step 1: Notch filter at 50 Hz ---
    b_notch, a_notch = iirnotch(w0=notch_freq, Q=quality, fs=fs)
    lfp_notched = filtfilt(b_notch, a_notch, lfp)

    # --- Step 2: Bandpass filter between 0.1–100 Hz ---
    low, high = band
    if low <= 0:
        low = 0.1
    if high >= fs / 2:
        high = fs / 2 - 1

    sos = butter(4, [low, high], btype='band', fs=fs, output='sos')
    filtered_lfp = sosfiltfilt(sos, lfp_notched)
    lfp_ds = decimate(filtered_lfp, 5, ftype='fir')
    lfp_ds = decimate(lfp_ds, 6, ftype='fir')

    fs_ds = fs/30

    return lfp_ds, fs_ds

lfp_notch = filtfilt(b_notch,a_notch,lfp[:,51])
b_notch,a_notch = iirnotch(w0=50.,Q=30,fs=30000)