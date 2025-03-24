"""Example: Student MRP"""

GAMMA = 0.1


def get_return(sample):
    """Get the return"""
    gt = 0
    for time_step, st in enumerate(sample):
        gt += (GAMMA ** time_step) * reward[st]
    return gt


episodes = [
    [5, 3, 4, 2, 3, 2, 3],
]

reward = {
    1: 3.0,
    2: 8.0,
    3: 8.0,
    4: -3.0,
    5: 13.0,
}

for eps in episodes:
    sample_return = get_return(sample=eps)
    print(f"{eps}: {round(sample_return, 3)}")
