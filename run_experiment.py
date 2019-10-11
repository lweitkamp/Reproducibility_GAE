import argparse
import os
import sys
from VPG import train

# TO SPLIT UP THE COMPUTE, REMOVE THE RETURNS YOU ARE NOT GOING TO RUN
# IF INTERUPTED: REMOVE THE learning_rates AND n_steps YOU HAVE DONE ALREADY TO CONTINUE WHERE YOU LEFT
learning_rates = [0.0001, 0.0003, 0.0005, 0.0009, 0.001, 0.003, 0.005, 0.007, 0.009, 0.01]
n_steps = [1, 10, 20, 30, 40, 50, 100, 150, 200]
seeds = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
returns = ["GAE", "Q", "A"]
OUTPUT_FOLDER = "results"

def write_to_file(filepath, results):
    with open(filepath, 'w') as f:
        for r in results:
            f.write(str(r))
            f.write("\n")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vanilla Policy Gradient Training')
    parser.add_argument('--env', type=str, default='CartPole-v0', help='gym environment name')
    parser.add_argument('--num_envs', type=int, default=16, help='number of parallel environments to run')
    parser.add_argument('--num_steps', type=int, default=20, help='number of steps the agent takes before updating')
    parser.add_argument('--lr', type=float, default=3e-2, help='Learning rate for optimizer')
    parser.add_argument('--max_steps', type=int, default=50000, help='maximum number of steps to take in the env')
    parser.add_argument('--test_every', type=int, default=1000, help='get testing values')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.97, help='GAE lambda, variance adjusting parameter')
    parser.add_argument('--return_function', type=str, default="GAE", help='The returns function we use from {GAE, Q, A}')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')

    ARGS = parser.parse_args()

    OUTPUT_FOLDER = os.path.join(os.getcwd(), OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if ARGS.seed != -1:
        print("if you do a manual run make sure to fill seed, return_function, lr, num_steps")
        os.makedirs(os.path.join(OUTPUT_FOLDER, str(ARGS.seed)), exist_ok=True)
        output_folder = os.path.join(OUTPUT_FOLDER, str(ARGS.seed), ARGS.return_function)
        os.makedirs(output_folder, exist_ok=True)

        filename = f"{ARGS.num_steps}_{ARGS.lr}.txt"
        filepath = os.path.join(output_folder, filename)
        print("ARGS", ARGS)
        results = train(ARGS)
        write_to_file(filepath, results)

    else:
        # commence disgusting combinatoric loop
        for seed in seeds:
            ARGS.seed = seed
            output_folder = os.path.join(OUTPUT_FOLDER, str(seed))
            os.makedirs(output_folder, exist_ok=True)

            for rf in returns:
                ARGS.return_function = rf
                rf_output_folder = os.path.join(output_folder, rf)
                os.makedirs(rf_output_folder, exist_ok=True)

                for num_steps in n_steps:
                    ARGS.num_steps = num_steps
                    tmp_output_folder = os.path.join(rf_output_folder, str(num_steps))
                    os.makedirs(tmp_output_folder, exist_ok=True)
                    for lr in learning_rates:
                        ARGS.lr = lr
                        filename = f"{ARGS.lr}.txt"
                        filepath = os.path.join(tmp_output_folder, filename)
                        results = train(ARGS)
                        write_to_file(filepath, results)

    print("done")
