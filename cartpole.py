# https://stable-baselines3.readthedocs.io/en/master/guide/rl.html
# https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html#a-taxonomy-of-rl-algorithms
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

import os
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

"""
### Description

This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
    in the left and right direction on the cart.

### Action Space

The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
    of the fixed force the cart is pushed with.

| Num | Action                 |
|-----|------------------------|
| 0   | Push cart to the left  |
| 1   | Push cart to the right |

**Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
    the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

### Observation Space

The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

| Num | Observation           | Min                 | Max               |
|-----|-----------------------|---------------------|-------------------|
| 0   | Cart Position         | -4.8                | 4.8               |
| 1   | Cart Velocity         | -Inf                | Inf               |
| 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
| 3   | Pole Angular Velocity | -Inf                | Inf               |

**Note:** While the ranges above denote the possible values for observation space of each element,
    it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
-  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
    if the cart leaves the `(-2.4, 2.4)` range.
-  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
    if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

### Rewards

Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
including the termination step, is allotted. The threshold for rewards is 475 for v1.

### Starting State

All observations are assigned a uniformly random value in `(-0.05, 0.05)`

### Episode End

The episode ends if any one of the following occurs:

1. Termination: Pole Angle is greater than ±12°
2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
3. Truncation: Episode length is greater than 500 (200 for v0)


### Run in terminal to visualize the training:
tensorboard --logdir={log_path}

"""

# ---------- Simple test and Visualize ---------------------------------
env = gymnasium.make("CartPole-v1", render_mode="human")
print(env.action_space.sample()) # 0-push cart to left, 1-push cart to the right
print(env.observation_space.sample()) # [cart position, cart velocity, pole angle, pole angular velocity]

episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info, _ = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

# -------------- Define Vectorised Environment --------------------------
vec_env = make_vec_env("CartPole-v1", n_envs=4)

model_path = os.path.join('./', 'Training', 'Saved_Models')
model_path_x = os.path.join('./', 'Training', 'Saved_Models', 'PPO_cartpole')
best_model_path = os.path.join('./', 'Training', 'Saved_Models', 'best_model')
log_path = os.path.join('./', 'Training', 'Logs')

# -------------- Training ------------------------------------------------
print("Starting the training")
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=50000)
del model

# -------------- Evaluation -----------------------------------------------
model = PPO.load(model_path_x, vec_env=vec_env)
print(evaluate_policy(model, vec_env, n_eval_episodes=10, render=True))
vec_env.close()

episodes = 5
for episode in range(1, episodes+1):
    obs = vec_env.reset()
    dones = np.zeros(4)
    score = 0
    while not dones.any():
        vec_env.render("human")
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        score+=rewards[0]
    print('Episode:{} - Score:{}'.format(episode, score))
vec_env.close()


# ------ Adding a callback to the training stage -------------------------

print("Starting the training with callbacks")
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
eval_callback = EvalCallback(vec_env, 
                             callback_on_new_best=stop_callback, 
                             eval_freq=10000, 
                             best_model_save_path=model_path, 
                             verbose=1)
model = PPO('MlpPolicy', vec_env, verbose = 1, tensorboard_log=log_path)
model.learn(total_timesteps=100000, callback=eval_callback)
del model

model = PPO.load(best_model_path, vec_env=vec_env)
print(evaluate_policy(model, vec_env, n_eval_episodes=10, render=True))
vec_env.close()

# -------------- Custom Policies/Architecture ---------------------------
net_arch=dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128])
model = PPO('MlpPolicy', vec_env, verbose = 1, policy_kwargs={'net_arch': net_arch})
model.learn(total_timesteps=100000, callback=eval_callback)


# -------------- Using an Alternate Algorithm ----------------------------
print("Starting the training with DQN algorithm")
model = DQN('MlpPolicy', vec_env, verbose = 1, tensorboard_log=log_path)
dqn_path = os.path.join('./', 'Training', 'Saved_Models', 'DQN_cartpole')

model.learn(total_timesteps=100000, callback=eval_callback)
model = DQN.load(dqn_path, vec_env=vec_env)

print(evaluate_policy(model, vec_env, n_eval_episodes=10, render=True))
vec_env.close()