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
    
    # Check if CatBot caught the cat
    if done:
        # Max reward for catching
        return 100.0
    
    # Decode positions
    # Getting CatBot's position (r_b, c_b)
    r_b = state // 1000
    c_b = (state % 1000) // 100
    # Getting the Cat's position (r_c, c_c)
    r_c = (state % 100) // 10
    c_c = state % 10
    
    # Get CatBot's next position
    r_nb = next_state // 1000
    c_nb = (next_state % 1000) // 100
    # Get Cat's next position
    r_nc = (next_state % 100) // 10
    c_nc = next_state % 10

    # Use Manhattan distance to get distance to cat
    dist_to_cat = abs(r_c - r_b) + abs(c_c - c_b)
    next_dist_to_cat = abs(r_nc - r_nb) + abs(c_nc - c_nb)
    
    # Reward for getting closer to the cat (Proximity Reward)
    # Give a small reward for decreasing the distance, and a small
    # penalty for increasing it.
    
    reward = 0.0
    if next_dist_to_cat < dist_to_cat:
        reward += 1.0 # Closer
    elif next_dist_to_cat > dist_to_cat:
        reward -= 1.0 # Farther

    # Penalty for every step taken
    reward -= 0.2
    
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
    
    # Hyperparameters
    learning_rate = 0.1
    discount_factor = 0.9
    
    # Exploration Strategy (Epsilon-Greedy)
    epsilon = 1.0            # Initial exploration rate
    max_epsilon = 1.0        # Max exploration rate
    max_epsilon = 0.01       # Minimum exploration rate
    # Decay rate
    decay_rate = 0.0001
    
    # Step counter limit in case of loop
    max_steps = 100 # Slightly higher than evaluation limit for training

    
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
        
        while not done and step < max_steps:
            # Explore or exploit (Epsilon-Greedy Strategy)
            if random.uniform(0, 1) < epsilon:
                # Explore: choose a random action
                action = env.action_space.sample()
            else:
                # Exploit: choose the action with the max Q-value
                action = np.argmax(q_table[current_state])

            # Take the action and observe
            # The environment returns: observation, reward, terminated, truncated, info 
            next_observation, _, terminated, truncated, info = env.step(action)
            next_state = next_observation
            
            # Terminated or truncated to determine if the episode is over
            done = terminated or truncated

            # Compute reward using helper function
            reward = compute_reward(current_state, next_state, done)

            # Update Q-table
            # Get the current Q(s, a) value
            current_q = q_table[current_state][action]
            
            # Get the max Q(s', a') value (next max Q-value)
            next_max_q = np.max(q_table[next_state])
            
            # Calculate new Q-value
            new_q = current_q + learning_rate * (reward + discount_factor * next_max_q - current_q)
            
            # Update the Q-table
            q_table[current_state][action] = new_q
            
            # Move to the next state
            current_state = next_state
            step += 1
            
        # Update Epsilon after the episode ends
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