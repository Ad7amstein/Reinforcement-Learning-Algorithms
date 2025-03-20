"""Action preferences script"""

import numpy as np


def softmax(preferences: np.ndarray) -> np.ndarray:
    """
    Compute the softmax probabilities over actions given preference values.
    """
    exp_prefs = np.exp(preferences - np.max(preferences))  # stability trick
    return exp_prefs / np.sum(exp_prefs)


def update_baseline(baseline: float, reward: float, alpha: float) -> float:
    """
    Update the single scalar baseline (running average of rewards).
    baseline_{t+1} = baseline_t + beta * (reward - baseline_t)
    """
    return baseline + alpha * (reward - baseline)


def update_preferences(
    preferences: np.ndarray,
    probs: np.ndarray,
    chosen_action: int,
    step_size: float,
    reward: np.ndarray,
    base_line: float,
) -> np.ndarray:
    """Action numerical preference updating"""
    arr_cpy = preferences.copy()
    for i, _ in enumerate(arr_cpy):
        if i == chosen_action:
            print(arr_cpy[i], step_size, reward, base_line, probs[i])
            arr_cpy[i] = arr_cpy[i] + step_size * (reward - base_line) * (
                1.0 - probs[i]
            )
        else:
            arr_cpy[i] = arr_cpy[i] - step_size * (reward - base_line) * probs[i]

    return arr_cpy


actions = np.array([5, 3, 4, 5, 2, 5])
rewards = np.array([0.37, -0.97, -0.26, 0.36, -1.05, 0.58])
action_preferences = np.array([0.21, 0.18, 1.98, 1.03, 0.39])
action_probs = np.zeros_like(action_preferences)
# n_actions = {1: 10, 2: 10, 3: 10, 4: 10}
# base_line_rewards = {1: 0.49, 2: 0.49, 3: 0.49, 4: 0.49}
ALPHA = 0.75
BASE_LINE = 0.76
N = 7

print(action_probs)

for step in range(6):
    # 1. Selected Action
    action = int(actions[step]) - 1
    # 2. Calculate action probabilities
    action_probs = softmax(action_preferences)
    action_probs = np.round(action_probs, 2)  # Round
    # 3. Update preferences
    action_preferences = update_preferences(
        action_preferences, action_probs, action, ALPHA, rewards[step], BASE_LINE
    )
    action_preferences = np.round(action_preferences, 2)  # Round

    # base_line_rewards[action] = updated_q
    # n_actions[action] += 1
    # Update Base Line
    
    # U4. pdate Base Line and N
    N += 1
    BASE_LINE = update_baseline(BASE_LINE, rewards[step], ALPHA)
    BASE_LINE = round(BASE_LINE, 2)  # Round
    print(f"BASE_LINE: {BASE_LINE}")
    print(f"action_probs: {action_probs}")
    print(f"action_preferences: {action_preferences}\n")
