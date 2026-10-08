import numpy as np

def band_integral(frequency, psd, low, high):
    f=np.asarray(frequency); p=np.asarray(psd)
    return float(p[(f>=low)&(f<high)].sum()*(f[1]-f[0]))

def resting_cli(theta, alpha, rest_theta, rest_alpha):
    r=np.log(np.asarray(rest_theta)/rest_alpha)
    center=np.median(r); scale=1.4826*np.median(abs(r-center))
    if scale<1e-6: raise ValueError('Degenerate resting baseline')
    return np.maximum((np.log(np.asarray(theta)/alpha)-center)/scale,0.)

def paired_mae(reference, estimate):
    return float(np.mean(abs(np.asarray(estimate)-reference)))
