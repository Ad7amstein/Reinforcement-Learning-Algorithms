"""Example: value update using bellman equation"""

import numpy as np

GAMMA = 0.13

def get_state_value(state, old_values, rs, ps):
    """Update state value using bellman equation"""
    state -= 1
    vs = rs[state]
    for s_num, s_ps in enumerate(ps[state]):
        vs += GAMMA * s_ps * old_values[s_num]

    return vs

transition_matrix = np.array([
    [0.00, 0.00, 0.42, 0.00, 0.58],
    [0.00, 0.37, 0.00, 0.25, 0.38],
    [1.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.12, 0.00, 0.88, 0.00],
    [0.00, 0.00, 0.00, 0.26, 0.74]
])

reward = np.array([-1.0, 13.0, -3.0, 8.0, 12.0])

value = np.array([39.0, 6.0, -3.0, 5.0, 35.0])

state5_value = get_state_value(5, old_values=value, rs=reward, ps=transition_matrix)
print(f"State 5 Value: {round(state5_value, 3)}")
