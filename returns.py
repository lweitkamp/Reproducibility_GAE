"""
This file implements various return methods such as

- (n-step) Monte Carlo return
- Generalized Advantage Estimation (GAE)


High-Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
https://arxiv.org/abs/1506.02438
"""

import torch

def returns():
    raise NotImplementedError

def GAE(next_value, rewards, masks, values, args):
    """
    Calculate the Generalized Advantage Estimation return as proposed in Schulman et al. 2015

    :param next_value: value estimation at timestep t+1
    :param rewards: rewards for each timestep
    :param masks: masks for env multiprocessing
    :param values: state value estimates for each timestep
    :param args: argument list
    :return: Return the 'returns', the discounted sum of rewards at each timestep
    """
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + args.gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + args.gamma * args.gae_lambda * masks[step] * gae
        returns.insert(0, gae + values[step])

    returns = torch.stack(returns).detach()
    returns = (returns - returns.mean()) / (returns.std() + 1e-10)
    return returns
