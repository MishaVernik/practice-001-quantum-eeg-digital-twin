"""Use student functions in an inference-result adapter."""
import argparse,importlib,json
from pathlib import Path
import numpy as np
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--solution',action='store_true');ap.add_argument('--out',type=Path,default=Path('student_analysis.json'));args=ap.parse_args()
    if args.out.exists():raise SystemExit('Choose a new output file.')
    m=importlib.import_module('solutions' if args.solution else 'tasks')
    raw=json.loads((Path(__file__).resolve().parent/'evidence/predictions.json').read_text())
    p=np.array(raw['probabilities']['born_gray_entangled']);y=np.array(raw['truth'])
    samples=m.sample_levels(p,np.random.default_rng(20261007))
    high=m.high_level_probability(p)
    result={'synthetic':True,'horizon':'one observed transition','nll':m.negative_log_likelihood(p,y),
        'example':{'day':raw['day'][0],'probabilities':p[0].tolist(),'probability_levels_6_7':float(high[0]),'sampled_level':int(samples[0])},
        'mean_probability_levels_6_7':float(np.mean(high))}
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
