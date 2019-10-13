# Measuring the Effect of Advantage Estimation in Actor-Critic methods

For the reinforcement learning course at the UvA we are given the task of creating reproducible research, where we can choose from a number of topics.
We have chosen to focus on $n-$step bootstrapping in actor-critic methods, where in specific we want to focus on variance reduction methods such
as the advantage estimation and the generalized advantage estimation (GAE), which naturally leads to the following question:

*What is the effect of (generalized) advantage estimation on the return in $n$-step bootstrapping?*

This blog post is meant to answer that question. First, we will give a quick overview of actor critic methods, $n$-step bootstrapping and the (generalized) advantage function. Second, we will discuss our setup which involves a detailed explanation of how the results can be reproduced. Third, we will give an analysis of the results which should answer the question above.

## Actor Critic Methods and Advantage Estimation

In reinforcement learning we typically want to maximize the total expected reward, which can be done using various methods. We can for example choose to learn the value function(s) for each state and infer the policy from this, or learn the policy directly through parameterization of the policy. Actor critic methods combine the two approaches: the actor is a parameterized policy, the critic learns the value for each state through bootstrapping[^4]:
$$
\begin{align*}
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})&= \mathbb{E}\left[ \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta (a_t|s_t) \big(r_{t} + \gamma \hat{v}(s_{t+1})  \big)\right] \\
&= \mathbb{E}\left[ \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta (a_t|s_t)\hat{q}(s_t, a_t)\right]
\end{align*}
$$
This estimate is however biased (due to bootstrapping) and can exhibit high variance (a common problem in policy gradient based methods). Bias is hard to tackle, but we can reduce variance through the introduction of the advantage function $\hat{A}$:
$$
\begin{align*}
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})&= \mathbb{E}\left[ \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta (a_t|s_t) \hat{A}\right] \\
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})&= \mathbb{E}\left[ \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta (a_t|s_t) \big(\hat{q}(s_t, a_t) - \hat{v}(s_t))\right]
\end{align*}
$$
The Advantage function tells us how good an action $a$ was in state $s$ compared to other actions we could have taken, so it tells us what the *advantage* of taking this action. If the action taken lead to a high return, we would like to increase the probability of taking that action! 

### N-step bootstrapping

1. explain n-step bootstrapping
2. how does it affect the return (monte-carlo returns are unbiased, but are they better?)

### Generalized Advantage Estimation

We now turn to an idea proposed in the paper *High Dimensional Continuous Control Using Generalized Advantage Estimation*[^1] by Schulman et al. 2016. The advantage function reduces variance, but the authors claim we can do better: learn an exponentially-weighted estimator of the advantage function. This is what they call Generalized Advantage Estimation

1. some math
2. thorough explanation of parameters i guess?

## Setup 

In our experiment we performed a grid search over the learning rate and the n-step return. 
For gamma we took 0.99 and for the lambda in GAE we took 0.97. Schulman et al. [^1]  show that these values work best with the GAE. We experiment with two other targets as well, these are the $n$-step Q-value and $n$-step Advantage. The Q-value is defined as $r_{t+1} + \gamma v_w (s_{t+1})$ and the Advantage is $r_{t+1} + \gamma v_w (s_{t+1}) - v_w(s)$. These returns are calculated for $n$-steps and normalised. 

TODO: explain how the returns are generalised to $n$-steps.

To ensure reproducability, we have manually set the seeds for _pytorch_ and the _gym_ environments. We use seeds [0, 30, 60, 90, 120] for _pytorch_ and the environments use seeds 16 to 31, where the seeds are determined by the number of workers (environment instances) we have. In our case this is 16, the seeds are calculated roughly as followed:

```seeds = [i + num_envs for i in range(num_envs)]``` 

The learning rates we investigate are given in the following table:

| Return type | learning rates |
| ----------- | ---- |
| Q, A          | [0.0001, 0.0003, 0.0005, 0.0009, 0.001, 0.003, 0.005, 0.007, 0.009, 0.01]     |
| GAE           |[0.01, 0.03, 0.05, 0.09, 0.1]     |

Our experimental results are obtained by iterating over the pytorch seeds, after which we iterate over the number of steps and finally over the possible learning rates. We do this for every return type. 

Our code base can be found <a href="https://github.com/lweitkamp/Reproducibility_GAE">here</a>, which is inspired by other implementations[^2][^3], 



## Results and Analysis

To determine which setup works best, we first combine the results of all the seeds, sorted by return type, $n$ used in $n$-steps and learning rate. We then calculate the mean over the rewards, and use this to determine which the best setup per return type.

The results are displayed in the next table:





## Citations

[^1]:  Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*.
[^2]: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
[^3]: https://github.com/ikostrikov/pytorch-a3c
[^4]: Actually, bootstrapping is what defines actor critic methods when contrasted to vanilla policy gradient methods.


