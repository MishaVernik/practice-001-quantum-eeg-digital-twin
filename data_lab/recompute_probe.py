"""Recompute the included 30-second ECG/PPG window from actual recorded samples."""
from pathlib import Path
import json, numpy as np
from cardiac_reference import from_signal
root=Path(__file__).resolve().parent
raw=np.load(root/'data/raw_cardiac_probe.npz')
reference=json.loads((root/'data/cardiac_windows.json').read_text())[0]
assert raw['participant'].item()==reference['participant']
result={}
for key,signal,source in [('ecg','ecg','ecg_rr'),('prv_native','ppg','ppg_prv')]:
    row=from_signal(raw['time'],raw[signal],source,4.,34.)
    expected=reference[key]
    assert row['valid']==expected['valid']
    assert row['rejection_reason']==expected['rejection_reason']
    if row['valid']:
        for field in ['rate_bpm','rmssd_ms']:
            np.testing.assert_allclose(row[field],expected[field],atol=1e-9)
    result[key]=row
print(json.dumps(result,indent=2))
print('PASS: raw ECG/PPG probe agrees with the archived window, including PPG rejection')
