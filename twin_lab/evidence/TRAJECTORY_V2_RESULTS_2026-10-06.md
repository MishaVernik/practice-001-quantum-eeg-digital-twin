# State-trajectory comparison – software evidence

6 October 2026. **Synthetic only. No human prediction, quantum advantage, planning benefit or productivity improvement established.** No hardware or paid jobs.

Canonical run: [trajectory-v2-20261006-causal/results.json](trajectory-v2-20261006-causal/results.json).

## Design and result

15 generated days: eight training days, two model-selection days, five later test days. Eight training-fitted load bins; 885 eligible test transitions. The benchmark does not change CMEFlow's displayed CLI. All probabilities and artifact replay checks are retained.

The target is the next observed load level. Inputs are available at the **forecast origin**, not at the target observation. Recording gaps and bad-contact segments are not joined. Unknown break history remains unknown; gaps do not become recovery. This is neither an emotion model nor a validated action-effect simulator.

| Model | Test negative log-likelihood (lower is better) |
|---|---:|
| Persistence | 2.5433 |
| Task-conditioned marginal | 1.6204 |
| Context-conditioned Markov | 1.1485 |
| Ridge autoregression | 0.9415 |
| Regularized softmax | **0.7881** |
| Small MLP | 0.8346 |
| GRU | 1.0152 |
| Born / Gray / product | 2.0078 |
| Born / Gray / entangled | 1.3408 |
| Born / binary / product | 1.9050 |
| Born / binary / entangled | 1.1966 |

Softmax outperformed ridge here: mean test NLL difference about −0.1534; paired five-day bootstrap interval [−0.2050, −0.0986]. The best Born arm did not beat ridge: its corresponding interval is [0.0647, 0.4454]. These intervals describe this generated fixture, not people or independent repeated experimental runs.

No quantum benefit was demonstrated. Eight epochs and one seed are a smoke comparison, not exhaustive tuning or a general rejection of QML. One MLP search candidate reached its iteration limit; that warning is not hidden or treated as proof of convergence. The GRU has longer history than the feedforward/Born arms, so its comparison is a stronger-history baseline, not an identical-memory claim.

## Corrections and provenance

- `trajectory-v2-20261005`: diagnostic first run; superseded by continuity corrections.
- `trajectory-v2-20261006-corrected`: continuity fixed, but target-window context still used. Superseded; do not cite as causal forecasting evidence.
- `trajectory-v2-20261006-causal`: origin-only context, duplicate-session refusal and explicit finite/unknown break checks. Use this run for the table above.

The old folders remain intact for traceability. The canonical folder contains the frozen configuration, code hashes, day splits, predictions, per-day results, paired uncertainty and all 11 serialized models. `deployment_allowed` is false in every artifact. Only load trusted project-created pickle artifacts.

## Reproduction

From the repository root, using a **new** output directory:

```powershell
.\flow-classifier\.venv\Scripts\python.exe -B validation\benchmark_trajectory_v2.py --synthetic --epochs 8 --out docs\evidence\state_planning\trajectory-v2-reproduction
.\flow-classifier\.venv\Scripts\python.exe -B -m unittest discover -s validation -p test_trajectory_v2.py -v
```

Wall time, artifact bytes, fitted-parameter counts where implemented and stored numeric scalars are in the JSON. Missing fitted-count methods are reported as null, not zero; storage and serialized size remain reported. Runtime measurements are local CPU measurements and not clean cross-hardware latency benchmarks.

## Still required

Real wearer data with auditable task/break/outcome timing; full matched-history comparisons; multi-seed and order-insensitive controls; finite-shot/noise tests; multi-step trajectory calibration; CP-SAT deterministic scheduling reference; optimizer/outcome comparisons; cost-accounted hardware experiments only with separate approval. Rank real plans only after held-out outcome improvement. A lower predicted CLI is not the planning objective.
