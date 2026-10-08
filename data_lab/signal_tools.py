"""Small explicit version of the project's offline EEG filter and PSD contract."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch

def filtered_signal(signal, fs):
    x=np.asarray(signal,dtype=float)
    x=sosfiltfilt(butter(4,0.5/(fs/2),btype='highpass',output='sos'),x)
    for hz in (50.,100.):
        if hz < (fs/2)*.98:
            b,a=iirnotch(hz/(fs/2),30.)
            x=filtfilt(b,a,x)
    return x

def band_power(signal, fs, low, high):
    x=filtered_signal(signal,fs)
    f,p=welch(x,fs=fs,nperseg=len(x),detrend='constant')
    return float(p[(f>=low)&(f<high)].sum()*(f[1]-f[0]))
