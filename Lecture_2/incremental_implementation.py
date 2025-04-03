"""Incremental implementation of action value 
estimation"""


def Q_a(q_a, r_i, n):
    """action value estimation"""
    if n == 0:
        return 0
    return q_a + (1/n) * (r_i - q_a)

actions = [
    1.00, 3.00, 3.00, 2.00, 1.00,
    3.00, 3.00, 2.00, 2.00, 3.00, 1.00,
    1.00, 3.00, 3.00, 3.00, 2.00, 1.00, 2.00, 2.00, 3.00
]

rewards = [
    -0.59, -0.39, -0.79, -0.90, 0.77, -0.16,
    -1.02, -0.16, 0.89, 0.76, -1.01, -1.26,
    0.24, 0.03, -1.61, -1.34, -0.12, 1.82, -0.42, -1.39
]


n_action = {1: 7, 2: 3, 3: 5}
q_action = {1: 0.98, 2: 0.90, 3: 2.77}

for i in range(20):
    action = int(actions[i])
    updated_q = Q_a(q_action[action], rewards[i], n_action[action])
    q_action[action] = round(updated_q, 2)
    n_action[action] += 1
    print(q_action)
