"""
To clean up the main files (VPG, PPO, etc.) this file contains some helper functions
"""
from torch.distributions import Categorical
import torch


def test(env, policy):
    with torch.no_grad():
        obs = torch.from_numpy(env.reset())

        done = False
        total_reward = 0
        while not done:
            probs, _ = policy(obs)
            action = Categorical(probs).sample()
            obs, reward, done, _ = env.step(action.item())
            total_reward += reward
            obs = torch.from_numpy(obs)
    return torch.tensor(total_reward, requires_grad=False)