import datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack  # , GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from actions import ACTIONS
from agent import Mario
from constants import *
from metrics import MetricLogger
from wrappers import ResizeObservation, SkipFrame, GrayScaleObservation

# to increase learning process, off rendering
# todo: test it
render = True
render = 'human' if render else None

# Initialize Super Mario environment
if gym.__version__ < '0.26':
    # old version, without render parameter,
    # need to use env.render() in code
    env = gym_super_mario_bros.make(
        id=level,
        new_step_api=True
    )
else:
    env = gym_super_mario_bros.make(
        id=level,
        render_mode=render,
        apply_api_compatibility=True
    )

# Limit the action-space to
#   0. walk right
#   1. jump right
#   todo: why? sometimes tasks can be solved faster with other actions,
#    but ofc, less action-space, less brain is needed
env = JoypadSpace(env, ACTIONS)

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

# to start model from some checkpoint, use it
checkpoint = None  # Path('checkpoints/2023-10-01T19-04-15/mario_net_1.chkpt')
mario = Mario(
    inputs=(4, 84, 84),
    actions=env.action_space.n,
    save_dir=save_dir,
    checkpoint=checkpoint
)

logger = MetricLogger(save_dir)

# from guide - 40000 (20 hrs on GPU (c) author)
# 50h on MPS (Apple Silicon M2 Pro (2023),
# main bad thing here, is that MPS allow usage
# only of 32-bit types variables (like float32, etc))
episodes = 40_000

# for Loop that train the model num_episodes times by playing the game
for e in range(episodes):
    state = env.reset()
    # Play the game!
    while True:
        # Run agent on the state
        action = mario.act(state)
        # Agent performs action
        next_state, reward, done, trunc, info = env.step(action)
        # Remember
        mario.cache(state, next_state, action, reward, done)
        # Learn
        q, loss = mario.learn()
        # Logging
        logger.log_step(reward, loss, q)
        # Update state
        state = next_state
        # Check if end of game
        if done or info["flag_get"]:
            break

    logger.log_episode()
    if e % 10 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
        print('explored: ', mario.explored)
