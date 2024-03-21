from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

delta_plot = []
iterations = 0

DECK_SIZE = 25
SKIPS = 4
EXPLODING_KITTEN = 1
INITIAL_DEFUSE = 1
INITIAL_STATE = (INITIAL_DEFUSE, 4, DECK_SIZE)

V = {}
for defuse in range(2):
    for skip in range(SKIPS + 1):
        for deck_size in range(1, DECK_SIZE + 1):
            V[(defuse, skip, deck_size)] = 0


def transition_and_reward_corrected_without_ek(state, action):
    defuse, skip, deck_size = state
    transitions = []

    ek_prob = EXPLODING_KITTEN / deck_size
    skip_prob = max(SKIPS-skip, 0) / deck_size
    cat_prob = 1 - ek_prob - skip_prob

    if action == 'draw':
        new_deck_size = max(deck_size - 1, 1)
        if defuse > 0:
            transitions.append(((max(defuse - 1, 0), skip, new_deck_size), ek_prob, -40))
        else:
            transitions.append(((defuse, skip, new_deck_size), ek_prob, -100))

        if skip < SKIPS:
            transitions.append(((defuse, skip + 1, new_deck_size), skip_prob, 50))
        transitions.append(((defuse, skip, new_deck_size), cat_prob, 25))

    elif action == 'skip' and skip > 0:
        transitions.append(((defuse, max(skip - 1, 0), deck_size), 1, -20))

    return transitions

gamma = 0.9
theta = 0.1

for defuse in range(3):
    for skip in range(SKIPS + 1):
        for deck_size in range(1, DECK_SIZE + 1):
            V[(defuse, skip, deck_size)] = 0

states_change = defaultdict(list)
while True:
    delta = 0
    for state in V.keys():
        states_change[state].append(V[state])
        v = V[state]
        V[state] = max(sum(p * (reward + gamma * V.get(next_state, 0)) for next_state, p, reward in transition_and_reward_corrected_without_ek(state, action)) for action in ['skip', 'draw'] if not (action == 'skip' and state[1] == 0))
        delta = max(delta, abs(v - V[state]))
    print(delta)
    delta_plot.append(delta)
    iterations += 1
    if delta < theta:
        break


plt.figure(figsize=(8, 4))
plt.plot(range(1, iterations + 1), delta_plot)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Delta', fontsize=12)
plt.title('Value Iteration Convergence: Delta vs. Iterations', fontsize=14)
plt.grid(True, linestyle="--")
plt.show()

optimal_policy = {}

state_ = []
draw = []
skip = []

print(V.keys())
for state in V.keys():
    best_action_value = float('-inf')
    best_action = None
    possible_actions = []
    for action in ['skip', 'draw']:
        if action == 'skip' and state[1] == 0:
            continue

        action_value = sum(p * (reward + gamma * V.get(next_state, 0)) for next_state, p, reward in transition_and_reward_corrected_without_ek(state, action) for action in ['skip', 'draw'] if not (action == 'skip' and state[1] == 0))
        possible_actions.append((action, action_value))

        if action_value > best_action_value:
            best_action_value = action_value
            best_action = action

    print(state, possible_actions)
    if(state[0] == 0 and state[1] == 1):
      skip.append(possible_actions[0][1])
      draw.append(possible_actions[1][1])

    optimal_policy[state] = best_action


optimal_policy_subset = {k: optimal_policy[k] for k in list(optimal_policy.keys())}


plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(draw)), draw, label="Draw", marker='o', markersize=2)
plt.plot(np.arange(len(draw)), skip, label="Skip", marker='o', markersize=2)

plt.xlabel("Deck Size", fontsize=12)
plt.ylabel("Value Function", fontsize=12)
plt.title("Value Function Comparison when defuse=0 and skip=1", fontsize=14)
plt.xticks(np.arange(len(draw)))
plt.axvline(x=4, color='g', linestyle='--', linewidth=2)
plt.legend(fontsize=12)

plt.show()
