"""Check student functions against data, plus units and boundary conditions."""
from pathlib import Path
import argparse, importlib, csv, json
import numpy as np
from scipy.signal import welch
from signal_tools import filtered_signal
from run_real import cli

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',action='store_true');a=ap.parse_args()
    student=importlib.import_module('solutions' if a.reference else 'tasks')
    root=Path(__file__).resolve().parent
    raw=np.load(root/'data/raw_eeg_probe.npz');names=raw['channel_names'].tolist();fs=float(raw['sfreq'])
    sig=filtered_signal(raw['MATBeasy'][names.index('FZ')],fs)
    freq,psd=welch(sig,fs=fs,nperseg=len(sig),detrend='constant')
    value=student.band_integral(freq,psd,4,8)
    np.testing.assert_allclose(value,psd[(freq>=4)&(freq<8)].sum()*.5)
    # One bin at each boundary: the upper edge belongs to the next band.
    np.testing.assert_allclose(student.band_integral(np.arange(10.),np.ones(10),4,8),4.)
    rows=list(csv.DictReader((root/'data/cli_bands.csv').open(encoding='utf-8')))
    base=[r for r in rows if r['subject']=='P01' and r['session']=='S1' and r['condition']=='base']
    task=[r for r in rows if r['subject']=='P01' and r['session']=='S1' and r['condition']=='lv0']
    def power(rr,key):return np.array([float(r[key]) for r in rr])
    th='theta_frontal_uv2';al='alpha_posterior_uv2'
    args=[power(task,th),power(task,al),power(base,th),power(base,al)]
    np.testing.assert_allclose(student.resting_cli(*args),cli(*args),atol=1e-10)
    try:student.resting_cli(np.ones(1),np.ones(1),np.ones(30),np.ones(30))
    except ValueError:pass
    else:raise AssertionError('A constant baseline must be rejected')
    pairs=[r for r in json.loads((root/'data/cardiac_windows.json').read_text()) if r['ecg']['valid'] and r['prv_native']['valid']]
    ref=np.array([r['ecg']['rate_bpm'] for r in pairs]);est=np.array([r['prv_native']['rate_bpm'] for r in pairs])
    np.testing.assert_allclose(student.paired_mae(ref,est),.25563478569642667,atol=1e-10)
    print(f'PASS: real EEG theta power {value:.6f} microvolt^2; CLI and 58 paired rates verified')

if __name__=='__main__':main()
