import numpy as np
import random
from cvxopt import matrix, solvers


U = np.array([[1, 0.5],  
              [0.5, 1]]) 

c = matrix(-U.flatten())  


G = matrix(-np.eye(4)) 
h = matrix(np.zeros(4)) 

A = matrix([[1.0, 1.0, 0.0, 0.0],  
            [0.0, 0.0, 1.0, 1.0]]) 
b = matrix([1.0, 1.0])  


solvers.options['show_progress'] = False
solution = solvers.lp(c, G, h, A.T, b)


p_values = np.array(solution['x']).reshape(2,2)

print("Optimal probabilities P(s_p, m):")
print(p_values)





deck_size = 25
num_skips = 4
num_defuse = 2
num_exploding_kittens = 1
num_players = 2
player_states = [{'defuse': num_defuse, 'skip': num_skips} for _ in range(num_players)]


def estimate_loss(player_state, deck_size, action):
    if(deck_size==0):
      return -100
    if action == 'draw':
        return num_exploding_kittens / deck_size
    elif action == 'skip':
        return 0


def oco_decision(player_state, deck_size):
    if player_state['defuse'] > 0:
        return 'draw'
    elif player_state['skip'] > 0:
        return 'skip'
    return 'draw'

def update_state(player_state, action, deck_size):
    if action == 'draw':
        card = random.choice(['card']* (deck_size - num_exploding_kittens) + ['exploding_kitten'])
        deck_size -= 1
        if card == 'exploding_kitten':
            player_state['defuse'] -= 1  
    elif action == 'skip':
        player_state['skip'] -= 1
    return deck_size


T = 100  
regret = [0 for _ in range(num_players)]
for t in range(1, T+1):
    for player_idx in range(num_players):
        player_state = player_states[player_idx]
        action = oco_decision(player_state, deck_size)
        loss = estimate_loss(player_state, deck_size, action)
        deck_size = update_state(player_state, action, deck_size)
        regret[player_idx] += loss - estimate_loss(player_state, deck_size, 'skip')

average_regret = [r / T for r in regret]
print("Average Regret per player:", average_regret)
