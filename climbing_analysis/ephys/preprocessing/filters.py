from scipy.signal import butter, filtfilt, sosfiltfilt, iirnotch, decimate, tf2sos, resample_poly
import numpy as np

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

def design_bandpass_sos(fs, band=(0.1, 100.0), order=4):
    low, high = band
    low = max(low, 0.1)
    high = min(high, fs/2 - 1.0)
    return butter(order, [low, high], btype="band", fs=fs, output="sos")

def design_notch_sos(fs, notch_freq=50.0, Q=30.0):
    b, a = iirnotch(w0=notch_freq, Q=Q, fs=fs)
    return tf2sos(b, a)

def downsample_lfp_fast(
    lfp, fs_in,
    fs_mid=1000, fs_out=200,
    notch_freq=50.0, Q=30.0,
    band=(0.1, 100.0),
    chunk_s=30,
):
    """
    lfp: np.ndarray [C, N] float32/float64
    Returns: lfp_out [C, N_out], fs_out
    """
    assert lfp.ndim == 2
    lfp = lfp.T
    C, N = lfp.shape

    # --- Stage 1: Downsample 30k -> 1k (polyphase) ---
    # Choose rational up/down for resample_poly
    # For 30000->1000: up=1, down=30
    up1, down1 = int(fs_mid), int(fs_in)
    # simplify ratio by gcd
    g1 = np.gcd(up1, down1)
    up1 //= g1
    down1 //= g1

    # --- Stage 2: Notch + bandpass at fs_mid ---
    sos_notch = design_notch_sos(fs_mid, notch_freq=notch_freq, Q=Q)
    sos_bp = design_bandpass_sos(fs_mid, band=band, order=4)

    # --- Stage 3: Downsample 1k -> 200 (polyphase) ---
    up2, down2 = int(fs_out), int(fs_mid)
    g2 = np.gcd(up2, down2)
    up2 //= g2
    down2 //= g2

    # Chunking (in input samples). Chunk *before* stage1.
    chunk_N = int(chunk_s * fs_in)

    out_chunks = []
    for start in range(0, N, chunk_N):
        stop = min(N, start + chunk_N)
        x = lfp[:, start:stop]

        # 1) resample to fs_mid
        x_mid = resample_poly(x, up=up1, down=down1, axis=1).astype(np.float32, copy=False)

        # 2) notch + bandpass (zero-phase)
        x_mid = sosfiltfilt(sos_notch, x_mid, axis=1)
        x_mid = sosfiltfilt(sos_bp, x_mid, axis=1)

        # 3) resample to fs_out
        x_out = resample_poly(x_mid, up=up2, down=down2, axis=1).astype(np.float32, copy=False)

        out_chunks.append(x_out)

    lfp_out = np.concatenate(out_chunks, axis=1)
    return lfp_out, fs_out