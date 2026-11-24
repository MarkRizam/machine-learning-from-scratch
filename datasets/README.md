# Dataset provenance

All experiments generate synthetic arrays locally using NumPy's `default_rng` with seeds 11, 22, 33. There are no downloaded datasets, personal identifiers, participant records or separately licensed datasets committed. See the exact generator in `src/ml_from_scratch/experiments.py` and the distributions and splits in `research/protocol.md`.

Each supervised run uses 200 independent 3-feature rows, split in generation order into 120 training, 40 validation and 40 test observations. Parameter sweeps use validation only; no hyperparameter selection on test data. Regression adds independent Gaussian noise, classification thresholds a noisy linear score. PCA is trained on the training split and evaluated by reconstruction on the test split. Clustering uses a separate 90-point synthetic training set; inertia is a within-sample objective. No real-world generalization or participant-related claim is made. Notebook examples are separate synthetic demonstrations.
