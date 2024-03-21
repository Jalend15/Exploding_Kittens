import numpy as np
import random
class TwoPlayerExplodingKittensEnvWithAttack:
    def __init__(self):
        self.deck_size = 25
        self.skip_cards = 4
        self.attack_cards = 2 
        self.exploding_kittens = 1
        self.defuse_cards = 2
        self.reset()

    def reset(self):
        self.players_state = [{'defuse': 1, 'skip': 2, 'attack': 0}, {'defuse': 1, 'skip': 2, 'attack': 0}]
        self.deck = ['card'] * (self.deck_size - self.skip_cards - self.attack_cards - self.exploding_kittens - self.defuse_cards * 2) + \
                    ['skip'] * self.skip_cards + \
                    ['attack'] * self.attack_cards + \
                    ['exploding_kitten'] * self.exploding_kittens + \
                    ['defuse'] * self.defuse_cards * 0
        random.shuffle(self.deck)
        self.current_player = 0
        self.turns_remaining = 1 
        self.game_over = False
        return self.get_state(self.current_player)

    def get_state(self, player_id):
        return (self.players_state[player_id]['defuse'], self.players_state[player_id]['skip'], self.players_state[player_id]['attack'])

    def step(self, player_id, action):
        reward = 0
        if action == 'draw':
            if self.turns_remaining > 0:
                self.turns_remaining -= 1
                card_drawn = self.deck.pop() if self.deck else 'no_cards_left'
            if card_drawn == 'exploding_kitten':
                if self.players_state[player_id]['defuse'] > 0:
                    self.players_state[player_id]['defuse'] -= 1
                    self.deck.append('exploding_kitten')
                    random.shuffle(self.deck)
                    reward = 5  
                else:
                    self.game_over = True
                    reward = -100  
            elif card_drawn == 'skip':
                self.players_state[player_id]['skip'] += 1
                reward = 10 
            elif card_drawn == 'defuse':
                self.players_state[player_id]['defuse'] += 1
                reward = 100  
            elif card_drawn == 'no_cards_left':
                self.game_over = True
                reward = 0  
            elif card_drawn == 'attack':
                self.players_state[player_id]['attack'] += 1
                reward = 1  
            else:
                reward = 0
                
        elif action == 'skip' and self.players_state[player_id]['skip'] > 0:
            self.players_state[player_id]['skip'] -= 1
            reward = 50 
        
        elif action == 'attack' and self.players_state[player_id]['attack'] > 0:
            self.players_state[player_id]['attack'] -= 1
            self.turns_remaining = 0 
            self.current_player = 1 - self.current_player  
            self.turns_remaining += 2 
            reward = 10  
        else:
            reward = -10 
        if self.turns_remaining == 0:  
            self.current_player = 1 - self.current_player
            self.turns_remaining = 1 
            
        if not self.deck:  
            self.game_over = True
            reward += 100

        return self.get_state(player_id), reward, self.game_over


class TwoPlayerQLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.3):
        self.q_table = [{}, {}] 
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_values(self, player_id, state):
        if state not in self.q_table[player_id]:
            self.q_table[player_id][state] = {'draw': 0, 'skip': 0, 'attack': 0}
        return self.q_table[player_id][state]

    def choose_action(self, player_id, state):
        if random.uniform(0, 1) < self.epsilon:
            possible_actions = ['draw']
            if state[1] > 0:  
                possible_actions.append('skip')
            if state[2] > 0:  
                possible_actions.append('attack')
            return random.choice(possible_actions)
        else:
            q_values = self.get_q_values(player_id, state)
            return max(q_values, key=q_values.get)

    def update_q_values(self, player_id, state, action, reward, next_state):
        future_rewards = max(self.get_q_values(player_id, next_state).values())
        current_q = self.get_q_values(player_id, state)[action]
        self.q_table[player_id][state][action] = current_q + self.alpha * (reward + self.gamma * future_rewards - current_q)

def train_two_player_agent_with_attack(episodes=10000):
    env = TwoPlayerExplodingKittensEnvWithAttack()
    agents = [TwoPlayerQLearningAgent(), TwoPlayerQLearningAgent()]

    for episode in range(episodes):
        state = env.reset()
        while not env.game_over:
            current_player = env.current_player
            state = env.get_state(current_player)
            action = agents[current_player].choose_action(current_player, state)
            next_state, reward, done = env.step(current_player, action)
            agents[current_player].update_q_values(current_player, state, action, reward, next_state)
            if done:
                break 

    return agents[0].q_table, agents[1].q_table

q_table_player1, q_table_player2 = train_two_player_agent_with_attack(episodes=1000)
print("Player 1 Q-Table:", q_table_player1)
print("Player 2 Q-Table:", q_table_player2)
