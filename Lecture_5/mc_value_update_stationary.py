"""Monte Carlo State Value Update Example in stationary environment"""

import numpy as np

TERMINAL_STATE = 3
GAMMA = 0.45

episode = np.array([5, 1, 5, 1, 5, 2, 1, 3])
state_rewards = np.array([5.00, 4.00, 0.00, 5.00, -11.00])
state_values = np.array([-30.00, -18.00, 0.00, -3.00, -10.00])
state_visits = np.array([17, 17, 10, 16, 19])


def update_state_value(state, values, st_return, st_visits):
    """Updates state value based on Monte Carlo Update rule"""
    return values[state] + (1 / st_visits[state]) * (st_return - values[state])


def get_returns(episode_, rewards):
    """Calculates Return for each state in episode"""
    steps = episode_.shape[0]
    ep_return = np.zeros_like(episode_, dtype=float)
    state_returns = np.zeros_like(rewards)
    for step in range(steps - 1, -1, -1):
        cur_state = episode_[step] - 1
        ep_return[step] += rewards[cur_state]
        if step < steps - 1:
            ep_return[step] += ep_return[step + 1] * GAMMA
        state_visits[cur_state] += 1
        state_values[cur_state] = update_state_value(
            cur_state, state_values, ep_return[step], state_visits
        )
        state_returns[cur_state] += ep_return[step]

    return ep_return, state_returns


def main():
    """Main Program"""
    episode_returns, state_returns = get_returns(episode, state_rewards)
    print(f"Return per state: {np.round(episode_returns, 2)}")

    # for state in range(0, 5):
    #     state_values[state] = update_state_value(
    #         state, state_values, state_returns, state_visits
    #     )

    print(f"State Values Updated: {np.round(state_values, 2)}")

    print(f"Update Number of Visits: {state_visits}")


if __name__ == "__main__":
    main()
