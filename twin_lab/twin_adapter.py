"""Explicit observation/forecast boundary for a teaching digital twin."""
import numpy as np

class TwinEstimator:
    def estimate(self,*,domain,source_id,observation,born,classical,quantum_available):
        result={'domain':domain,'source_id':source_id,'observation':observation,
                'forecast':None,'mode':'observation_only','planning_allowed':False}
        if domain!='synthetic_trajectory':
            if born is not None or classical is not None:
                raise ValueError('No validated transition model for this observation domain')
            result['reason']='No matched transition model';return result
        p=born if quantum_available and born is not None else classical
        if p is None:result['reason']='No available model';return result
        p=np.asarray(p,dtype=float)
        if p.shape!=(8,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1):
            raise ValueError('Expected eight finite nonnegative probabilities summing to one')
        result['mode']='born_simulator' if quantum_available and born is not None else 'classical_fallback'
        result['forecast']={'horizon':'one observed transition','probabilities':p.tolist(),
                            'expected_level':float(p@np.arange(8)),'p_levels_6_7':float(p[6:].sum())}
        return result
