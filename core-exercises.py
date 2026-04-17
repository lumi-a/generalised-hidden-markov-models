import numpy as np


def run(T, initial, ws, phi=None) -> float:
    """Returns the probability of seeing observations ws[0],ws[1],... in the GHHM given
    by T and the initial distribution.
    T has shape (observation, previous_state, next_state).
    If phi is none, assumes a standard HMM.
    """
    if phi is None:
        phi = np.ones(T.shape[1])

    matrix_product = np.eye(T.shape[1], T.shape[2])
    for w in ws:
        matrix_product = matrix_product @ T[w]
    return initial @ matrix_product @ phi


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
xor_initial = np.array([1, 0, 0, 0, 0])
run_xor = lambda ws: run(xor_T, xor_initial, ws)

for a in [0, 1]:
    for b in [0, 1]:
        expected = 1 if a != b else 0
        assert run_xor([a, b, expected]) == 0.25
        assert run_xor([a, b, 1 - expected]) == 0
