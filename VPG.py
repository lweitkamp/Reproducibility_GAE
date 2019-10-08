import argparse
from torch.distributions import Categorical
import numpy as np
import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from wrappers import make_env, SubprocVecEnv
from models import ActorCriticMLP
from utils import test
from returns import GAE


def policy_gradient(logp_actions, state_values, returns):
    logp_actions = torch.stack(logp_actions)
    state_values = torch.stack(state_values)

    advantage = returns - state_values

    actor_loss = -(logp_actions * advantage.detach())
    critic_loss = advantage.pow(2)
    loss = (actor_loss + critic_loss).sum()
    return loss


def train(args):
    print(args)
    envs = SubprocVecEnv([make_env(args.env, i) for i in range(args.num_envs)], args.MC)
    test_env = gym.make(args.env); test_env.seed(42)
    policy = ActorCriticMLP(input_dim=envs.observation_space.shape[0], n_acts=envs.action_space.n)
    optim = torch.optim.Adam(params=policy.parameters(), lr=args.lr)

    test_rewards = []
    steps = 1
    running_reward = 0

    obs = torch.from_numpy(envs.reset())
    while steps < args.max_steps:

        logp_actions = []
        state_values = []
        rewards      = []
        masks        = []

        for _ in range(args.num_steps):
            probs, state_value = policy.forward(obs)

            dist = Categorical(probs)
            action = dist.sample()

            obs, reward, done, _ = envs.step(action.numpy())

            logp_actions.append(dist.log_prob(action).unsqueeze(1))
            state_values.append(state_value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1))
            obs = torch.from_numpy(obs)
            steps += 1

            if steps % args.test_every == 0:
                test_reward = np.mean([test(test_env, policy) for _ in range(10)])
                test_rewards.append(test_reward)
                running_reward = 0.1 * test_reward + (1 - 0.1) * running_reward
                writer.add_scalar('rewards/test', float(running_reward), steps)
                print(f"Running reward at timestep {steps}: {running_reward}. and {test_reward}")

            if (1-done).sum() == 0:
                break

        _, next_value = policy(obs)
        returns = GAE(next_value, rewards, masks, state_values, args)
        loss = policy_gradient(logp_actions, state_values, returns)

        optim.zero_grad()
        loss.backward()
        optim.step()
        # if monte carlo, reset after
        if args.num_steps == 200:
            obs = torch.from_numpy(envs.reset())



parser = argparse.ArgumentParser(description='Vanilla Policy Gradient Training')
parser.add_argument('--env', type=str, default='CartPole-v0', help='gym environment name')
parser.add_argument('--num_envs', type=int, default=16, help='number of parallel environments to run')
parser.add_argument('--num_steps', type=int, default=20, help='number of steps the agent takes before updating')
parser.add_argument('--lr', type=float, default=3e-2, help='Learning rate for optimizer')
parser.add_argument('--max_steps', type=int, default=100000, help='maximum number of steps to take in the env')
parser.add_argument('--test_every', type=int, default=1000, help='get testing values')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--gae_lambda', type=float, default=0.97, help='GAE lambda, variance adjusting parameter')
parser.add_argument('--MC', type=bool, default=False, help='Match n-steps with the number of steps of termination')

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    ARGS = parser.parse_args()
    writer = SummaryWriter('runs')
    train(ARGS)
