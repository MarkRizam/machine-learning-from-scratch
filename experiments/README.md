# Experiment execution

Run `python -m ml_from_scratch.experiments` from the repository root. All eight experiment families are executed for seeds 11, 22 and 33, with 78 configuration/seed records. The script runs twice and checks numerical reproducibility at absolute tolerance 1e-10. See [protocol](../research/protocol.md) and [full results](../results/benchmarks.json). PNG and SVG figures are generated locally; no network data are used.
