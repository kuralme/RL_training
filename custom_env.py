import os
import numpy as np
import random
import gymnasium 
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# ---- Types of Spaces -----
# Discrete(3)
# Box(0,1,shape=(3,3)).sample()
# Box(0,255,shape=(3,3), dtype=int).sample()
# Tuple((Discrete(2), Box(0,100, shape=(1,)))).sample()
# Dict({'height':Discrete(2), "speed":Box(0,100, shape=(1,))}).sample()
# MultiBinary(4).sample()
# MultiDiscrete([5,2,2]).sample()Discrete(3)
# Box(0,1,shape=(3,3)).sample()
# Box(0,255,shape=(3,3), dtype=int).sample()
# Tuple((Discrete(2), Box(0,100, shape=(1,)))).sample()
# Dict({'height':Discrete(2), "speed":Box(0,100, shape=(1,))}).sample()
# MultiBinary(4).sample()
# MultiDiscrete([5,2,2]).sample()


class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # Temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set start temp
        self.state = 38 + random.randint(-3,3)
        # Set shower length
        self.shower_length = 60
        
    def step(self, action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0 
        # 2 -1 = 1 temperature 
        self.state += action -1 
        # Reduce shower length by 1 second
        self.shower_length -= 1 
        
        # Calculate reward
        if self.state >=37 and self.state <=39: 
            reward =1 
        else: 
            reward = -1 
        
        # Check if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = np.array([38 + random.randint(-3,3)]).astype(float)
        # Reset shower time
        self.shower_length = 60 
        return self.state

env = ShowerEnv()
print(env.observation_space.sample())
check_env(env, warn=True)

# Simple Test
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset(seed=None)
    done = False
    score = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

# Train and Evaluate
log_path = os.path.join('./', 'Training', 'Logs')
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=400000)
model.save('PPO')

print(evaluate_policy(model, env, n_eval_episodes=10, render=True))
env.close()
