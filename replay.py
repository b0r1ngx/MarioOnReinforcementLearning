import datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym.wrappers import FrameStack  # , GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from agent import Mario
from constants import *
from metrics import MetricLogger
from wrappers import ResizeObservation, SkipFrame, GrayScaleObservation

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

env = JoypadSpace(env, actions)

env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
# env = TransformObservation(env, f=lambda x: x / 255.)
if gym.__version__ < '0.26':
    env = FrameStack(env, num_stack=4, new_step_api=True)
else:
    env = FrameStack(env, num_stack=4)

env.reset()

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)

# prev:  # Path('checkpoints/2023-09-23T22-50-27/mario_net_8.chkpt')
checkpoint = Path('checkpoints/2023-09-26T17-03-14/mario_net_10.chkpt')
mario = Mario(
    state_dim=(4, 84, 84),
    action_dim=env.action_space.n,
    save_dir=save_dir,
    checkpoint=checkpoint
)
mario.exploration_rate = mario.exploration_rate_min

logger = MetricLogger(save_dir)

episodes = 400

for e in range(episodes):
    state = env.reset()
    while True:
        env.render()
        action = mario.act(state)
        next_state, reward, done, truncated, info = env.step(action)
        mario.cache(state, next_state, action, reward, done)
        logger.log_step(reward, None, None)
        state = next_state
        if done or info['flag_get']:
            break

    logger.log_episode()
    if e % 10 == 0:
        logger.record(
            episode=e,
            epsilon=mario.exploration_rate,
            step=mario.curr_step
        )
        print('explored: ', mario.explored)
