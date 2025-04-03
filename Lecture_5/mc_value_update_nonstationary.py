"""Monte Carlo State Value Update Example in non stationary environment"""

import numpy as np

TERMINAL_STATE = 3
GAMMA = 0.17
ALPHA = 0.03


def update_state_value(state, values, st_returns):
    """Updates state value based on Monte Carlo Update rule"""
    return values[state] + ALPHA * (st_returns[state] - values[state])

def get_returns(episode_, rewards):
    """Calculates Return for each state in episode"""
    steps = episode_.shape[0]
    ep_return = np.zeros_like(episode_, dtype=float)
    state_returns = np.zeros_like(rewards)
    for step in range(steps-1, -1, -1):
        cur_state = episode_[step]-1
        ep_return[step] += rewards[cur_state]
        if step < steps - 1:
            ep_return[step] += ep_return[step + 1] * GAMMA
        # state_values[cur_state] = update_state_value(cur_state, state_values, ep_return[step])
        state_returns[cur_state] += ep_return[step]

    return ep_return, state_returns

episode = np.array([4, 1, 6, 6, 5, 2, 3])
state_rewards = np.array([5.00, 1.00, 0.00, -1.00, 0.00, -2.00])
state_values = np.array([-17.00, -7.00, 0.00, -7.00, -12.00, -27.00])

def main():
    """Main Program"""
    episode_returns, state_returns = get_returns(episode, state_rewards)
    print(f"Episode Return: {np.round(episode_returns, 2)}")

    for state in range(0, 6):
        state_values[state] = update_state_value(state, state_values, state_returns)

    print(f"State Values Updated: {np.round(state_values, 2)}")

if __name__ == "__main__":
    main()
