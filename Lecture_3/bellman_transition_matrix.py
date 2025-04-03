"""Example: Get the transition matrix converting MDP to MP under a given policy"""

import numpy as np

def get_transition_matrix(policy_as, trans_mat_actions):
    """Get the transition matrix for states turning MDP to MP"""
    n_states = policy_as.shape[0]
    p_pi = np.zeros((n_states, n_states))

    # For each state, compute weighted sum of transition matrices
    for s in range(n_states):
        for a in range(policy_as.shape[1]):
            p_pi[s] += policy_as[s,a] * trans_mat_actions[a,s]

    return p_pi

policy = np.array([
    [0.19, 0.30, 0.19, 0.00, 0.32],
    [0.19, 0.00, 0.00, 0.61, 0.20],
    [0.06, 0.24, 0.31, 0.00, 0.39],
    [0.00, 0.82, 0.00, 0.18, 0.00]
])

P_a1 = [
    [0.41, 0.00, 0.50, 0.09],  # From S1
    [0.85, 0.07, 0.00, 0.08],  # From S2
    [0.38, 0.56, 0.06, 0.00],  # From S3
    [0.00, 0.00, 0.00, 1.00]   # From S4 (absorbing state)
]

P_a2 = [
    [0.00, 0.40, 0.28, 0.33],
    [0.00, 0.22, 0.78, 0.00],
    [0.00, 0.78, 0.00, 0.22],
    [0.07, 0.43, 0.42, 0.08]
]

P_a3 = [
    [0.00, 0.00, 0.00, 1.00],
    [0.00, 0.27, 0.00, 0.73],
    [0.00, 0.77, 0.23, 0.00],
    [0.00, 0.49, 0.00, 0.51]
]

P_a4 = [
    [0.00, 0.82, 0.18, 0.00],
    [0.25, 0.64, 0.00, 0.11],
    [0.83, 0.17, 0.00, 0.00],
    [0.00, 0.00, 0.00, 1.00]
]

P_a5 = [
    [0.92, 0.08, 0.00, 0.00],
    [0.00, 0.58, 0.38, 0.03],
    [0.00, 0.33, 0.32, 0.35],
    [0.00, 0.28, 0.31, 0.41]
]

P_actions = np.array([
    P_a1,
    P_a2,
    P_a3,
    P_a4,
    P_a5
])

trans_mat_states = get_transition_matrix(policy_as=policy, trans_mat_actions=P_actions)
print(trans_mat_states)
