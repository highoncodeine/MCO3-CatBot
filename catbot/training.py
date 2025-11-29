import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env

#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

# Helper function for reward computation
def compute_reward(state: int, next_state: int, done: bool) -> float:
    # State value is R_bot*1000 + C_bot*100 + R_cat*10 + C_cat
    
    # Check for terminal state (CatBot caught the cat)
    if done:
        # High positive reward for catching the cat
        return 100.0
    
    # Decode positions
    # CatBot position (r_b, c_b)
    r_b = state // 1000
    c_b = (state % 1000) // 100
    # Cat position (r_c, c_c)
    r_c = (state % 100) // 10
    c_c = state % 10
    
    # Next CatBot position (r_nb, c_nb)
    r_nb = next_state // 1000
    c_nb = (next_state % 1000) // 100
    # Next Cat position (r_nc, c_nc)
    r_nc = (next_state % 100) // 10
    c_nc = next_state % 10

    # Calculate Manhattan distance (distance to cat)
    dist_to_cat = abs(r_c - r_b) + abs(c_c - c_b)
    next_dist_to_cat = abs(r_nc - r_nb) + abs(c_nc - c_nb)
    
    # Reward for getting closer to the cat (Proximity Reward)
    # Give a small positive reward for decreasing the distance, and a small
    # negative reward (penalty) for increasing it.
    
    reward = 0.0
    if next_dist_to_cat < dist_to_cat:
        reward += 1.0 # Closer
    elif next_dist_to_cat > dist_to_cat:
        reward -= 0.5 # Farther

    # Small penalty for every step taken (to encourage catching the cat quickly)
    reward -= 0.1
    
    return reward


#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    # 1. Hyperparameters
    learning_rate = 0.1      # Alpha (α)
    discount_factor = 0.9    # Gamma (γ)
    
    # 2. Exploration Strategy (Epsilon-Greedy)
    epsilon = 1.0            # Initial exploration rate
    max_epsilon = 1.0        # Max exploration rate
    max_epsilon = 0.01       # Minimum exploration rate
    # Decay rate (must be tuned to decay over 5000 episodes)
    decay_rate = 0.0001
    
    # 3. Step counter limit for safety/efficiency
    MAX_STEPS_PER_EPISODE = 100 # Slightly higher than evaluation limit for training

    
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
               
        # 1. Reset the environment and get the initial state
        # The environment returns observation, info
        # The observation is the state represented as an integer
        observation, info = env.reset()
        current_state = observation
        
        # Tracking terminal condition
        done = False
        step = 0
        
        while not done and step < MAX_STEPS_PER_EPISODE:
            # 2. Decide whether to explore or exploit (Epsilon-Greedy Strategy)
            if random.uniform(0, 1) < epsilon:
                # Explore: choose a random action
                action = env.action_space.sample()
            else:
                # Exploit: choose the action with the max Q-value
                action = np.argmax(q_table[current_state])

            # 3. Take the action and observe the next state
            # The environment returns: observation, reward, terminated, truncated, info
            # Our environment always returns reward=0, so we ignore it. 
            next_observation, _, terminated, truncated, info = env.step(action)
            next_state = next_observation
            
            # Combine terminated and truncated to determine if the episode is over
            done = terminated or truncated

            # 4. Compute reward manually
            reward = compute_reward(current_state, next_state, done)

            # 5. Update the Q-table
            # Get the current Q(s, a) value
            current_q = q_table[current_state][action]
            
            # Get the max Q(s', a') value (next max Q-value)
            next_max_q = np.max(q_table[next_state])
            
            # Calculate the new Q-value
            new_q = current_q + learning_rate * (reward + discount_factor * next_max_q - current_q)
            
            # Update the Q-table
            q_table[current_state][action] = new_q
            
            # Move to the next state
            current_state = next_state
            step += 1
            
        # Update Epsilon (decay the exploration rate) after the episode ends
        epsilon = max_epsilon + (max_epsilon - max_epsilon) * np.exp(-decay_rate * ep)
        
        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table