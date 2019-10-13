import argparse
from torch.distributions import Categorical
import numpy as np
import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from wrappers import make_env, SubprocVecEnv
from models import ActorCriticMLP
from utils import test
from returns import GAE, Q, A


def policy_gradient(logp_actions, returns):
    logp_actions = torch.stack(logp_actions)

    actor_loss = -(logp_actions * returns.detach())
    critic_loss = returns.pow(2)
    loss = (actor_loss + critic_loss).sum()
    return loss


def train(args):
    print(args)
    writer = SummaryWriter('runs')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.return_function == "GAE":
        return_function = GAE
    elif args.return_function == "Q":
        return_function = Q
    elif args.return_function == "A":
        return_function = A

    MONTE_CARLO = True if args.num_steps == 200 else False

    envs = SubprocVecEnv([make_env(args.env, i + args.num_envs) for i in range(args.num_envs)], MONTE_CARLO)
    test_env = gym.make(args.env); test_env.seed(args.seed + args.num_envs)
    policy = ActorCriticMLP(input_dim=envs.observation_space.shape[0], n_acts=envs.action_space.n)
    optim = torch.optim.Adam(params=policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    test_rewards = []
    steps = 1

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
            # try:
            #     action = dist.sample()
            # except:
            #     print(obs, state_value, probs)

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
                writer.add_scalar('rewards/test', float(test_reward), steps)
                print(f"Running reward at timestep {steps}: and {test_reward}")

            if (1-done).sum() == 0:
                break

        next_value = 0
        if not (1-done).sum() == 0:
            _, next_value = policy(obs)

        returns = return_function(next_value, rewards, masks, state_values, args)
        loss = policy_gradient(logp_actions, returns)

        optim.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(policy.parameters(), 10)

        optim.step()
        # if monte carlo, we need to reset the environment by hand
        if MONTE_CARLO:
            obs = torch.from_numpy(envs.reset())
    return test_rewards

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vanilla Policy Gradient Training')
    parser.add_argument('--env', type=str, default='CartPole-v0', help='gym environment name')
    parser.add_argument('--num_envs', type=int, default=16, help='number of parallel environments to run')
    parser.add_argument('--num_steps', type=int, default=20, help='number of steps the agent takes before updating')
    parser.add_argument('--lr', type=float, default=3e-2, help='Learning rate for optimizer')
    parser.add_argument('--max_steps', type=int, default=100000, help='maximum number of steps to take in the env')
    parser.add_argument('--test_every', type=int, default=1000, help='get testing values')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.97, help='GAE lambda, variance adjusting parameter')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay')
    parser.add_argument('--return_function', type=str, default="GAE", help='The returns function we use from {GAE, Q, A}')

    ARGS = parser.parse_args()
    train(ARGS)
