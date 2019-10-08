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

def Advantage(next_value, rewards, masks, values, args):
    values = values + [next_value]
    returns = []
    for step in reversed(range(len(rewards))):

        Qsa = rewards[step] + args.gamma * values[step + 1] * masks[step]
        Vs  = values[step]
        A = Qsa - Vs
        returns.insert(0, A)

    # Still normalize the returns
    returns = torch.stack(returns).detach()
    returns = (returns - returns.mean()) / (returns.std() + 1e-10)
    return returns

def Q(next_value, rewards, masks, values, args):
    values = values + [next_value]
    returns = []
    for step in reversed(range(len(rewards))):
        Qsa = rewards[step] + args.gamma * values[step + 1] * masks[step]
        returns.insert(0, Qsa)

    # Still normalize the returns
    returns = torch.stack(returns).detach()
    returns = (returns - returns.mean()) / (returns.std() + 1e-10)
    return returns

def GAE(next_value, rewards, masks, values, args):
    """
    Calculate the Generalized Advantage Estimation return as proposed in Schulman et al. 2015

    :param next_value: value estimation at timestep t+1 (used to estimate Qsa)
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

        Qsa = rewards[step] + args.gamma * values[step + 1] * masks[step]
        Vs  = values[step]
        delta = Qsa - Vs

        gae = delta + args.gamma * args.gae_lambda * masks[step] * gae
        returns.insert(0, gae + values[step])

    # Still normalize the returns
    returns = torch.stack(returns).detach()
    returns = (returns - returns.mean()) / (returns.std() + 1e-10)
    return returns
