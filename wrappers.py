import gym
import numpy as np
import torch
from torchvision import transforms as T
from gym.spaces import Box


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = (shape,)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8
        )

    def observation(self, observation):
        transforms = T.Compose([
            T.Resize(self.shape),
            T.Normalize(0, 255)
        ])
        observation = transforms(
            observation
        ).squeeze(0)
        return observation


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = .0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            if term or trunc:
                break
        return obs, total_reward, term, trunc, info


def permute_orientation(observation):
    # permute [H, W, C] array to [C, H, W] tensor
    observation = np.transpose(
        observation,
        (2, 0, 1)
    )
    observation = torch.tensor(
        observation.copy(),
        dtype=torch.float
    )
    return observation


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(
            low=0,
            high=255,
            shape=obs_shape,
            dtype=np.uint8
        )

    def observation(self, observation):
        observation = permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation
