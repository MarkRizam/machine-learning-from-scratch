# AIR-001 V1.1 — scientific audit

## Scope and evidence
Reviewed V1 baseline commit `2fc835628816769bee4a8b787208ec9b1bc10f1a`, the protocol, implementation, metrics, experiments and documentation. The [baseline GitHub Actions run](https://github.com/MarkRizam/machine-learning-from-scratch/actions/runs/45429324194) completed successfully on Ubuntu and Windows, including tests, all eight experiments and notebook execution.

## Independent numerical re-execution
Re-executed the experiment function locally with one numerical-library thread and compared against all 78 committed per-seed/configuration records. Absolute tolerance 1e-10, relative tolerance zero; maximum observed absolute difference zero, both before and after input-validation changes. This does not establish identical floating-point outputs on every platform.

## Defects and corrections
The initial audit reproduced acceptance of NaN labels in KNN, Gaussian NB, decision trees and random forests. Gaussian NB generated invalid means and variances. Non-integral and non-finite configuration values could also be accepted at construction. V1.1 rejects non-finite numeric and unsupported object labels, retains homogeneous string labels, validates integral parameters (excluding booleans), and rejects non-finite clustering tolerance. Added 36 regression cases; 73/73 local tests passed, Python compilation passed. One scikit-learn warning about the unregularized logistic configuration remains; it is not a numerical failure.

## Scientific interpretation
- The deterministic synthetic 120/40/40 supervised split is respected. No test-based hyperparameter selection was found in the experiment script. Clustering evaluates training inertia, not generalization.
- Reported sample standard deviations are based on three seeds and are descriptive, not confidence intervals.
- Forest voting and RNG, tree splitting, K-means initialization and stopping rules are not identical to scikit-learn. No superiority claim is supported.
- Gradient descent compares equal epochs, not equal update budgets.
- All observations remain educational reproductions, not novel or peer-reviewed findings.
- No external datasets, participant information or paid APIs are needed.

## Release limits
Docker and Ruff remain unexecuted locally. Runtime and memory benchmarks, wider dataset coverage, exotic labels and extreme numerical magnitudes remain outside the verified scope. The fresh V1.1 hosted CI run must pass before release. The previous V1 report and committed benchmark numbers are preserved; this document supplements them.

Reproduce: `python -m pytest -q`, then `python scripts/verify_committed_results.py`. The new CI step compares committed numerical records before generating fresh result files.
