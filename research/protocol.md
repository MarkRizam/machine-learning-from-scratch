# AIR-001 — exploratory protocol
Written before the benchmark runs (after implementation and initial correctness tests). This is educational reproduction, not a preregistered or novel study.

Question: how closely do transparent NumPy algorithms reproduce reference implementations in small controlled synthetic settings?

Use seeds 11, 22, 33. Generate 200 independent standard-normal 3-feature observations per seed; regression target 4 + X@[1,2,-3] + independent Gaussian noise (sd 0.5). Classification target is X0 + X1 + independent standard normal noise > 0. First 120 train, next 40 validation, last 40 test. No preprocessing fitted on test data. Hyperparameters below fixed before runs; parameter sweeps use validation only. No parameter selection or claims about unseen domains. Clustering uses 90 generated points in 3 groups (means -5,0,5 in two dimensions, sd 0.2); evaluate training inertia, not generalization. PCA fit training only, reconstruction on test.

- EXP-001: OLS, batch gradient descent, reference OLS; test MSE. Expect similar MSE after convergence.
- EXP-002: logistic and Gaussian NB, corresponding references; test accuracy, logistic log loss and F1. Expect approximate agreement; not necessarily identical.
- EXP-003: KNN k=1,5,9 and Euclidean/Manhattan; validation accuracy, compare reference.
- EXP-004: tree depths 1,3,5; train and validation accuracy, compare reference. Expect train accuracy generally nondecreasing, validation may vary.
- EXP-005: forests with 3,7 trees and 1 or 2 candidate features; validation accuracy, reference forest with same nominal settings. RNG and voting differ.
- EXP-006: random and k-means++ initialization; one restart, training inertia, reference with n_init=1. Compare variability; expect k-means++ often more reliable, not guaranteed.
- EXP-007: PCA 1,2 components; test reconstruction MSE and training explained variance; reference SVD. Expect agreement up to sign/rotation.
- EXP-008: least squares optimization with batch sizes 1,16,120 and learning rates 0.01,0.05; 100 epochs; final training objective, reference optimum. Expect convergence depends on update count and noise; epochs do not equal optimizer steps.

Run twice; compare numeric outputs excluding execution environment/timing (atol 1e-10, rtol 0). Report all seeds and configurations. Runtime is not measured in this first increment. Only synthetic data; no external dataset redistribution. Metrics use binary labels where applicable. No population-level inference from three seeds.
