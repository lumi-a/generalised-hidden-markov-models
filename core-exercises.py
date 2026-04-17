from collections.abc import Sequence
from functools import partial
from random import random

import numpy as np
import numpy.typing as npt


def calculate_probability(
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


def predict(
    T: npt.NDArray[np.float64],
    initial: npt.NDArray[np.float64],
    ws: Sequence[int],
    phi: npt.NDArray[np.float64] | None = None,
) -> tuple[Sequence[float], Sequence[float]]:
    """Make predictions about the current state of the model and the next observation."""

    if phi is None:
        phi = np.ones(T.shape[1])

    unnormalised_belief_state = np.linalg.multi_dot([initial, *[T[w] for w in ws]])
    normalisation_factor = float(unnormalised_belief_state @ phi)
    belief_state = unnormalised_belief_state / normalisation_factor

    observation_prediction = [belief_state @ Tw @ phi for Tw in T]

    return belief_state, observation_prediction


p = random()
q = random()
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
run_xor = partial(calculate_probability, xor_T, xor_initial)

z1r_T = np.array(
    [[[0, 1, 0], [0, 0, 0], [0.5, 0, 0]], [[0, 0, 0], [0, 0, 1], [0.5, 0, 0]]]
)
z1r_initial = np.array([1 / 3, 1 / 3, 1 / 3])
predict_z1r = partial(predict, z1r_T, z1r_initial)


a = 0.2
x = 0.15
b = (1 - a) / 2
y = 1 - 2 * x
ay = a * y
bx = b * x
by = b * y
ax = a * x
mess3_T = np.array(
    [
        [
            [ay, bx, bx],
            [ax, by, bx],
            [ax, bx, by],
        ],
        [
            [by, ax, bx],
            [bx, ay, bx],
            [bx, ax, by],
        ],
        [
            [by, bx, ax],
            [bx, by, ax],
            [bx, bx, ay],
        ],
    ]
)
mess3_initial = np.array([1 / 3, 1 / 3, 1 / 3])
predict_mess3 = partial(predict, mess3_T, mess3_initial)

if __name__ == "__main__":
    for a in [0, 1]:
        for b in [0, 1]:
            expected = 1 if a != b else 0
            a_prob = p if a == 1 else 1 - p
            b_prob = q if b == 1 else 1 - q
            assert run_xor([a, b, expected]) == a_prob * b_prob, f"p={p}, q={q}"
            assert run_xor([a, b, 1 - expected]) == 0, f"p={p}, q={q}"

    assert predict_z1r([0])[0].tolist() == [1 / 3, 2 / 3, 0]
    assert predict_z1r([1])[0].tolist() == [1 / 3, 0, 2 / 3]
    assert predict_z1r([0, 0])[0].tolist() == [0, 1, 0]
    assert predict_z1r([0, 1])[0].tolist() == [0, 0, 1]
    assert predict_z1r([1, 0])[0].tolist() == [1 / 2, 1 / 2, 0]
    assert predict_z1r([1, 1])[0].tolist() == [1, 0, 0]

    print("Tests passed!")

    import itertools

    import plotly.graph_objects as go

    words = list(itertools.product([0, 1, 2], repeat=9))
    belief_states = np.array([predict_mess3(list(w))[0] for w in words])

    fig = go.Figure(
        data=go.Scatter3d(
            x=belief_states[:, 0],
            y=belief_states[:, 1],
            z=belief_states[:, 2],
            mode="markers",
            marker=dict(size=2, opacity=0.5),
        )
    )
    fig.update_layout(
        title="Belief states after all 8-symbol words (alphabet {0,1,2})",
        scene=dict(xaxis_title="State 0", yaxis_title="State 1", zaxis_title="State 2"),
    )
    fig.show()
