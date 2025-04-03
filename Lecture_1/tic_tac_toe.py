"""State value update module"""

def update_prob(vs1, vs2, a):
    """Update state value"""
    return vs1 + a * (vs2 - vs1)

states = [0.500, 0.544, 0.597, 1.000]
ALPHA = 0.82

for i in range(len(states)-1):
    print(update_prob(states[i], states[i+1], ALPHA))
