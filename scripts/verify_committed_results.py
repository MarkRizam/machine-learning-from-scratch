"""Re-execute AIR-001 and compare with the recorded per-seed benchmark."""
import json
from pathlib import Path

import numpy as np
from ml_from_scratch.experiments import run

ROOT = Path(__file__).resolve().parents[1]


def main():
    saved = json.loads((ROOT / 'results' / 'benchmarks.json').read_text())['records']
    current = run()
    assert len(saved) == len(current) == 78
    max_delta = 0.0
    for expected, actual in zip(saved, current):
        assert expected.keys() == actual.keys()
        for key, value in expected.items():
            if isinstance(value, float):
                np.testing.assert_allclose(actual[key], value, rtol=0, atol=1e-10)
                max_delta = max(max_delta, abs(actual[key] - value))
            else:
                assert actual[key] == value, (key, actual[key], value)
    print(f'78 committed records reproduced; max absolute delta={max_delta:.12g}')


if __name__ == '__main__':
    main()
