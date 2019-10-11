# Generalized Advantage Estimation

Generalized Advantage Estimation (GAE) is used in most state of the art policy gradient methods. In this blogpost we compare actor-critic methods and the use of (Generalized) Advantage Estimation to reduce the variance in gradient updates. 

In actor-critic methods, a baseline is used to reduce the variance of the gradient updates. Different functions for this baseline are possible, like the true value or action function. However we cannot always access these true value or action functions and thus we will estimate these as well. This however induces some bias in the gradient updates. Schulman et al. [^1] claim that their Generalized Advantage Estimation contains less variance than the estimated value and action functions. To check their claim we will use a vanilla policy gradient algorithm that uses the different baselines and compare their performances. 



## Policy gradients and GAE

Policy gradient methods maximize the expected total reward. They estimate the gradient, $g:=\nabla_\theta\mathbb{E}[\sum_{t=0}^\infty r_t]$, repeatedly to get the optimal policy. There exist different forms of the policy gradient where the $\Psi$ below can be filled in using different targets, like the total rewards of the episode, the rewards following action $a_t$, the state-action value, advantage function, etc.  








$$
g = \mathbb{E}\left[ \sum_{t=0}^\infty \Psi_t \nabla_\theta \log \pi_\theta (a_t|s_t) \right]
$$









## Setup 

In our experiment we performed a grid search over the learning rate and the n-step return. For gamma we took 0.99 and for the lambda in GAE we took 0.97. Schulman et al. [^1]  show that these values work best with the GAE. 



our code base can be found <a href="https://github.com/lweitkamp/Reproducibility_GAE">here</a>, which is inspired by other implementations[^2][^3], 





#### Citations 

[^1]:  Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*.
[^2]: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
[^3]: https://github.com/ikostrikov/pytorch-a3c