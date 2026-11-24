# Public AI Research Project 001 — Machine Learning From Scratch

**Status:** Experimental implementation; release review pending. Educational reproduction, not peer-reviewed research.

[GitHub repository](https://github.com/MarkRizam/machine-learning-from-scratch)

Why build machine learning algorithms from mathematical foundations? Implementing them reveals how models learn, where numerical errors occur and how evaluation can mislead. AIR-001 uses NumPy for nine algorithm families and scikit-learn solely as an independent reference. It complements RizamLearning's mathematics and machine learning lessons.

Eight controlled experiments across three seeds produced 78 configuration/seed records. Two runs matched at numerical absolute tolerance 1e-10. The OLS test MSE was 0.257542 for both implementations. Logistic test accuracy was 79.17% for both, and Gaussian NB test accuracy was 78.33%. PCA with two components achieved 0.331506 reconstruction MSE in both implementations. Tree and forest scores differed under some configurations; RNG, split choices and voting rules are not identical. These are synthetic demonstrations and do not establish performance on real data.

To reproduce: install Python 3.13, clone the reviewed source branch, run `python -m pip install -e ".[research]"`, `python -m pytest -q` and `python -m ml_from_scratch.experiments`. Nine Jupyter notebooks provide worked examples and visualizations. The local evaluation passed 37 tests and executed all nine notebooks; Docker and hosted CI require separate verification. The machine-readable results and protocol are included in the repository. Future work includes broader dataset comparisons and stronger edge-case coverage.
