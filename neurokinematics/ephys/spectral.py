def morlet_lfp(data,dflist,frame_captures,stances,node='r_hindpaw',epoch_loc='start',freqs=np.arange(2,40,1), n_cycles=None, xlim_=[-0.5,0.5], save_fig=None):
    erps,kinematics,fs = epoch_lfp(data,dflist,frame_captures,stances,node=node,epoch_loc=epoch_loc, xlim_=xlim_)
    lfp_epochs = erps.reshape([erps.shape[0],1,erps.shape[1]])
    
    if n_cycles == None:
        n_cycles = freqs/4
    power = mne.time_frequency.tfr_array_morlet(
        lfp_epochs,
        sfreq=fs,
        freqs=freqs,
        n_cycles=n_cycles,
        output='power'
    )
    # single channel, so remove index
    power = power.squeeze()
    # Average prestimulus baseline
    mu = power[:,:,int((fs*(np.abs(xlim_[0])-0.5))/2):int(fs*(np.abs(xlim_[0])-0.1))].mean(axis=2,keepdims=True)
    power_z = ((power-mu)/mu)*100
    #power_db = 10*np.log10(power_z)
    power_avg = power_z.mean(axis=0)
    times = np.linspace(xlim_[0],xlim_[1],int(fs*(np.sum(np.abs(xlim_)))))
    # Plot
    plt.figure(figsize=(8,4))
    plt.pcolormesh(times, freqs,power_avg, shading='gouraud', cmap='turbo')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Average Spectrogram: {node} {epoch_loc}  movement (n={str(power.shape[0])})')
    cbar = plt.colorbar()
    cbar.set_label('Relative Power (%)')
    plt.xlim(xlim_)
    if save_fig:
        plt.savefig(save_fig+'/morlet_example.pdf')
        plt.savefig(save_fig+'/morelet_example.png')
    plt.show()
    return power_z