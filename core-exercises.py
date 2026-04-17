from collections.abc import Sequence
from functools import partial

import numpy as np
import numpy.typing as npt


def run(
    T: npt.NDArray[np.float64],
    initial: npt.NDArray[np.float64],
    ws: Sequence[int],
    phi: npt.NDArray[np.float64] | None = None,
) -> float:
    """Returns the probability of seeing observations ws[0],ws[1],... in the GHHM given
    by T and the initial distribution.
    T has shape (observation, previous_state, next_state).
    If phi is None, assumes a standard HMM.
    """
    if phi is None:
        phi = np.ones(T.shape[1])
    return float(np.linalg.multi_dot([initial, *[T[w] for w in ws], phi]))


p = 0.5
q = 0.5
xor_T = np.array(
    [
        [
            [0, 1 - p, 0, 0, 0],
            [0, 0, 0, 1 - q, 0],
            [0, 0, 0, 0, 1 - q],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, p, 0, 0],
            [0, 0, 0, 0, q],
            [0, 0, 0, q, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ],
    ]
)
xor_initial = np.array([1, 0, 0, 0, 0], dtype=np.float64)
run_xor = partial(run, xor_T, xor_initial)

if __name__ == "__main__":
    for a in [0, 1]:
        for b in [0, 1]:
            expected = 1 if a != b else 0
            assert run_xor([a, b, expected]) == 0.25
            assert run_xor([a, b, 1 - expected]) == 0
