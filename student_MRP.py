"""Example: Student MRP"""

import numpy as np

GAMMA = 0.5


def get_return(sample):
    """Get the return"""
    gt = 0
    for time_step, st in enumerate(sample):
        gt += (GAMMA ** time_step) * reward[st]
    return gt

def get_value(state, returns):
    """Get the value of the state"""
    returns_np = np.array(returns)
    val = np.average(returns_np)
    return (state, val)

episodes = [
    ["C1", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "Sleep"],
    ["C1", "C2", "C3", "Pub", "C2", "C3", "Pass", "Sleep"],
    ["C1", "FB", "FB", "C1", "C2", "C3", "Pub", "C1", "FB",
     "FB", "FB", "C1", "C2", "C3", "Pub", "C2", "Sleep"]
]

reward = {
    "C1": -2,
    "C2": -2,
    "C3": -2,
    "FB": -1,
    "Pass": 10,
    "Sleep": 0,
    "Pub": 1
}

for eps in episodes:
    sample_return = get_return(sample=eps)
    print(f"{eps}: {round(sample_return, 3)}")
