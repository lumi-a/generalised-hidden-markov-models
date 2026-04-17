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

