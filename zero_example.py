import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import *
from nes_py.wrappers import JoypadSpace

from constants import level

if gym.__version__ < '0.26':
    env = gym_super_mario_bros.make(
        id=level,
        new_step_api=True
    )
else:
    env = gym_super_mario_bros.make(
        id=level,
        render_mode='human',
        apply_api_compatibility=True
    )

env = JoypadSpace(env, actions=SIMPLE_MOVEMENT)
env.reset()

done = True
while True:
    if done:
        state = env.reset()
    state, reward, done, trunc, info = env.step(env.action_space.sample())
    env.render()
