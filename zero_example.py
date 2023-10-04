import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

from actions import ACTIONS
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

env = JoypadSpace(env, actions=ACTIONS)
env.reset()
next_state, reward, term, trunc, info = env.step(action=0)
done = term or trunc
print(
    f"shape: {next_state.shape}, "
    f"reward: {reward}, "
    f"done: {done},\n"
    f"info: {info}"
)

done = True
while True:
    if done:
        state = env.reset()
    state, reward, term, trunc, info = env.step(env.action_space.sample())
    done = term or trunc
    env.render()
