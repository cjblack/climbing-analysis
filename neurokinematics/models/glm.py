import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln  # log gamma for log(y!)
from scipy.signal import decimate, savgol_filter
from sklearn.model_selection import KFold  # optional: requires scikit-learn; if unavailable, use custom CV
import statsmodels.api as sm
# -----------------------
# Utilities
# -----------------------
def get_trial_data(dflist, frame_captures, sorting, unit_id:int=19, trial:int=0, kin_range=[0,-1], type='velocity'):
    #spike_train = sorting.get_unit_spike_train(unit_id, return_times=True)
    spike_train = sorting.get_unit_spike_train(unit_id)/30000 # this makes it more friendly in terms of start time
    camera_frames = frame_captures[trial]
    df = dflist[trial]
    start_idx = len(camera_frames)-len(df)
    start_ts = camera_frames[start_idx]/30000 # start time of video acquisition in seconds
    end_ts = camera_frames[-1]/30000

    #spike_ts = spike_train - start_ts
    spike_ts = spike_train[(spike_train >= start_ts) & (spike_train <= end_ts)] # get only spike times within video acquisition window
    spike_ts = spike_ts - start_ts
    kinematics = make_kinematic_vector(df, type=type)
    if kin_range==[0,-1]:
        kinematics=kinematics
    else:
        kinematics=kinematics[kin_range[0]:kin_range[1],:]
    kinematics, spikes_binned, bin_width  = resample_data(kinematics, spike_ts)
    return kinematics, spikes_binned, bin_width
    
def create_and_plot_model(kinematics, spikes_binned, bin_width, model_type='poisson'):

    if model_type == 'poisson':
        glm_model, glm_results, predicted = create_glm(kinematics, spikes_binned)
        plot_glm_results(spikes_binned, predicted, bin_width)
    elif model_type == 'gaussian':
        glm_model, glm_results, predicted = create_glm(spikes_binned, kinematics)
        plot_glm_results(kinematics, predicted, bin_width)
    return glm_model, glm_results, predicted



def bin_spikes(spikes_ts, t_edges):
    """Bin spike timestamps into counts for bins defined by t_edges (len = T+1)."""
    counts, _ = np.histogram(spikes_ts, bins=t_edges)
    return counts  # length T

def make_time_edges(t0, t1, bin_size):
    """Create bin edges from t0 to t1 (seconds)."""
    return np.arange(t0, t1 + bin_size, bin_size)

def make_kinematic_vector(df,type='velocity'):
    """Returns an K x N vector of X any Y coordinates from kinematics"""
    kinematic = []
    for key in df.keys():
        if (key.startswith('r')) | (key.startswith('l')):
            kinematic.append(df[key].to_numpy())
    kinematic.append(df['snout_X'])
    kinematic.append(df['snout_Y'])
    kinematics = np.array(kinematic)
    if type == 'velocity':
        velocity=[]
        tvec = np.linspace(0,len(df)/200.,len(df))
        for i in range(kinematics.shape[0]):
            velocity.append(np.gradient(kinematics[i,:], tvec))
        velocity = np.array(velocity)
        kinematics = np.vstack([kinematics,velocity])
    return kinematics

def resample_data(kinematics, spike_ts, bin_width=0.2,fps=0.005):
    K,N = kinematics.shape
    max_time = N*fps
    start_time = 0
    end_time = np.ceil(max_time / bin_width) * bin_width
    bins = make_time_edges(start_time, end_time, bin_width)
    spikes_binned = bin_spikes(spike_ts, bins)
    factor = int(bin_width/fps)
    #kinematics_trimmed = kinematics[:, :kinematics.shape[1] - kinematics.shape[1] % factor]
    #kinematics_downsampled = kinematics_trimmed.reshape(K, -1, factor).mean(axis=2).T
    kinematics_downsampled  = decimate(kinematics,q=40,axis=1, ftype='fir',zero_phase=True)
    kinematics_downsampled = kinematics_downsampled.T
    min_time = min(spikes_binned.shape[0], kinematics_downsampled.shape[0])
    spikes_binned = spikes_binned[:min_time]
    kinematics = kinematics_downsampled[:min_time, :]
    return kinematics, spikes_binned, bin_width

def create_glm(kinematics, spikes_binned):
    """
    Poisson GLM for continuous spike + kinematic data
    - Input:
        spikes_ts: 1D array of spike timestamps in seconds (for *one* neuron)
        kin_ts: 1D array of timestamps for kinematic samples (seconds)
        kin: (N_kin, K) array of kinematic variables sampled at kin_ts
    - Output: fitted beta and evaluation metrics
    """
    X = sm.add_constant(kinematics)
    glm_poisson = sm.GLM(spikes_binned,X,family=sm.families.Poisson())
    glm_results = glm_poisson.fit()

    predicted_rate = glm_results.predict(X)
    return glm_poisson, glm_results, predicted_rate

def plot_glm_results(spikes_binned, predicted_rate, bin_width):
    time = np.arange(len(spikes_binned)) * bin_width

    # Null model prediction (mean spike count per bin)
    null_rate = np.full_like(spikes_binned, fill_value=np.mean(spikes_binned))

    # Optional: Smooth for visual clarity
    from scipy.signal import savgol_filter

    smooth_actual = savgol_filter(spikes_binned, window_length=11, polyorder=2)
    smooth_pred = savgol_filter(predicted_rate, window_length=11, polyorder=2)
    smooth_null = savgol_filter(null_rate, window_length=11, polyorder=2)
    # Calculate MSE
    mse = np.mean((predicted_rate-spikes_binned)**2)
    print(f'MSE: {mse}')
    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(time, smooth_actual, label="Actual", color='black', linewidth=1)
    plt.plot(time, smooth_pred, label="GLM Predicted", color='red', linestyle='--')
    plt.plot(time, smooth_null, label="Null Model", color='blue', linestyle=':')

    plt.xlabel("Time (s)")
    plt.ylabel("Spike count (smoothed)")
    plt.title(f"Actual vs. GLM vs. Null Model Prediction, MSE: {mse}")
    plt.legend()
    plt.tight_layout()
    plt.show()
