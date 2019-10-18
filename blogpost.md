# Measuring the Effect of Advantage Estimation in Actor-Critic methods
*By Noah van Grinsven, Anton Steenvoorden, Tessa Wagenaar, Laurens Weitkamp.*



Nowadays many of the RL policy gradient methods use Generalized Advantage Estimation (GAE) as a baseline in actor-critic methods. Schulman et al.[^1] state that GAE reduces the variance of the policy gradient estimate when compared to other baselines, like advantage estimation. This comes at a cost, because GAE can introduce a bias into this estimate. To check this we will focus on $n-$step bootstrapping in actor-critic methods, which traditionally exhibit high variance for higher $n$ and high bias for lower $n$. In specific, we want to compare variance reduction methods such as the advantage estimation and the generalized advantage estimation. This naturally leads to the following question:

*What is the effect of (generalized) advantage estimation on the return in $n$-step bootstrapping?*

This blogpost is meant to answer that question. Since GAE reduces the variance, we expect it will improve the performance in high variance problems, so in $n$-step bootstrapping with a high $n$. However in the case of $n$-step bootstrapping with a low $n$, the variance will not be very high, but we will have some bias. If we now apply GAE in this problem, we will have bias on top of bias and thus expect GAE to perform worse. 

First, we will give a quick overview of actor critic methods, $n$-step bootstrapping and the (generalized) advantage function. Second, we will discuss our setup which involves a detailed explanation of how the results can be reproduced. Third, we will give an analysis of the results which should answer the question above. Our code base can be found <a href="https://github.com/lweitkamp/Reproducibility_GAE">here</a>, which is inspired by other implementations[^2][^3].

## Actor Critic Methods and Advantage Estimation

In reinforcement learning we typically want to maximize the total expected reward, which can be done using various methods. We can for example choose to learn the value function(s) for each state and infer the policy from this, or learn the policy directly through parameterization of the policy. Actor-critic methods combine the two approaches: the actor is a parameterized policy whose output matches the number of actions ($a_t \in \mathcal{A}$), the critic learns the value for each state through bootstrapping[^4]. We can write the loss function to this objective as $J(\boldsymbol{\theta})$, where we use gradient descent to maximize this objective:
$$
\begin{align*}
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})&= \mathbb{E}\left[ \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta (a_t|s_t) \boldsymbol{\Psi}_t\right] \\
& \text{Where we have} \\
\boldsymbol{\Psi}_t&= r_t + \gamma r_{t+1} + \ldots + \gamma^{n-1} r_{t+n-1} + \gamma^n\hat{v}(s_{t+n})  - \hat{v}(s_t)
\end{align*}
$$
This choice of $\boldsymbol{\Psi}$ is commonly known as the advantage function. The Advantage function tells us how good an action $a$ was in state $s$ compared to how good the average of other actions are. So it tells us what the *advantage* of taking this action is. If the action taken in state $s$ leads to a high return, we would like to increase the probability of taking that action in state $s$.

This estimate can be biased (due to estimation and bootstrapping) and can exhibit high variance (a common problem in policy gradient based methods). 

### N-step bootstrapping

Monte Carlo based methods (such as actor critic) have one big disadvantage: we have to wait for the end of an episode to perform a backup. We can tackle this disadvantage by performing $n$-step bootstrapping with $n \in \mathbb{N}$. There is of course no free lunch, and performing bootstrapping does have the effect of (possibly) inducing bias to our estimate (this is especially true for $n$ close to 1). If we combine actor-critic methods with $n-$step learning and advantage estimation, it is commonly known as advantage actor critic (A2C). 

In code, estimating the advantage function through bootstrapping looks like this:

```python
def A(Rt, rewards, values, gamma): # R_t here is actually v(s_t), our bootstrapped estimate
    returns = []
    for step in reversed(range(len(rewards))):
        Rt = rewards[step] + gamma * Rt
        returns.insert(0, Rt)

    advantage = returns - values
    return advantage
```

Where the rewards and values are vectors of size $\mathbb{R}^n$, and hence we go through the list in reverse order to properly discount each reward.

### Generalized Advantage Estimation

We now turn to an idea proposed in the paper *High Dimensional Continuous Control Using Generalized Advantage Estimation*[^1] by Schulman et al. 2016. The advantage function reduces variance, but the authors claim we can use a better $\boldsymbol{\Psi}$ function: learn an exponentially-weighted estimator of the advantage function. This is what's called Generalized Advantage Estimation (GAE):
$$
\hat{A}^{\text{GAE}(\gamma, \lambda)}_t = \sum^{\infty}_{l=0} (\gamma \lambda)^{l} \delta^V_{t+l}
$$
Where $\delta^V_{t} = r_t + \gamma \hat{v}(s_{t+1}) - \hat{v}(s_t)$ is the bootstrapped estimate for $\hat{A}_t$. The reduction of variance is due to exponentially weighting of (partially, we also use the rewards at each timestep) bootstrapped estimates. The parameter $0 < \lambda < 1$ governs a trade-off between variance ($\lambda \approx 1$) and bias $(\lambda \approx 0)$. Note that this is ***bias on top of bias***! But the authors note that it is a bias we can permit, as it reduces the variance to such a degree to enable quick learning. Additionally, the authors note that it is desirable to set $\lambda << \gamma$ as to balance bias and variance. In code, it looks like this:

```python
def GAE(next_value, rewards, values, gamma, GAE_lambda):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        Qsa = rewards[step] + gamma * values[step + 1]
        Vs  = values[step]
        delta = Qsa - Vs

        gae = delta + gamma * GAE_lambda * gae
        returns.insert(0, gae)
    
    return returns
```

## Setup 

To answer our research question "*What is the effect of (generalized) advantage estimation on the return in $n$-step bootstrapping?*", we will vary the learning rate over different $n$-step bootstrapping methods. 
When we have found the optimal learning rates for the $n$-step bootstrapping methods, we will compare the results of these methods with each other to determine the performance of the Advantage and Generalized Advantage Estimation as critics in an actor-critic algorithm. 

### Environment

For our experiment we have chosen to use the CartPole-v0 environment of the OpenAI gym python package[^5].

The CartPole-v0 environment has two actions, namely push left and push right. The goal is to balance a pole on top of a cart (hence cartpole) for as long as possible (maximum of 200 time steps) and the input our agent receives is a vector of four values: pole position, cart velocity, pole angle and pole velocity at tip. A video of the environment with a random policy can be seen below on the left hand side.

<div style="width:100%; display:table;">
<!--<div style="float: left; width:49%">-->
<video controls autoplay loop="loop"  style="margin: 0 auto; width:50%"><source src="https://gym.openai.com/videos/2019-10-08--6QXvzzSWoV/CartPole-v1/thumbnail.mp4" type="video/mp4"></video>
<!--</div>
<div style="float: left; width:49%">
<video controls autoplay loop="loop" style="width:100%"><source src="https://gym.openai.com/videos/2019-10-08--6QXvzzSWoV/MountainCar-v0/thumbnail.mp4" type="video/mp4"></video> 
</div>-->
</div>

*Video was taken from [OpenAI Gym](https://gym.openai.com)* 

This environment was chosen for its simplicity, while still having a quite large state space. Additionally, CartPole-v0 was used in the original by Schulman et al. 2016[^1], although a different update method was used. More difficult environments have not been tested due to the limited time available for this project. 

### Actor-critic Implementation

We train the agent using a deep neural network where the input is transformed into shared features (a vector in $\mathbb{R}^{30}$), from which two heads form: the actor ($\in \mathbb{R}^{|\mathcal{A}|}$) and critic output ($\in \mathbb{R}$). A code snippet in PyTorch can be seen below. Note that the output for the actor is a Softmax.

```python
import torch.nn as nn

class ActorCriticMLP(nn.Module):
    def __init__(self, input_dim, n_acts, n_hidden=32):
        super(ActorCriticMLP, self).__init__()
        self.input_dim = input_dim
        self.n_acts = n_acts
        self.n_hidden = n_hidden

        self.features = nn.Sequential(
            nn.Linear(self.input_dim, self.n_hidden),
            nn.ReLU()
        )

        self.value_function = nn.Sequential(
            nn.Linear(self.n_hidden, 1)
        )
        self.policy = nn.Sequential(
            nn.Linear(self.n_hidden, n_acts),
            nn.Softmax(dim=0)
        )

    def forward(self, obs):
        obs = obs.float()
        obs = self.features(obs)
        probs = self.policy(obs)
        value = self.value_function(obs)
        return probs, value
```

In our experiment we performed a grid search over the learning rate and the n-step return. 
For gamma we took $0.99$ and for the lambda in GAE we took $0.97$, these values come recommended in the original paper for GAE[^1].

### Parameter search

The experiment first searches for the optimal learning rate for each $n$-step. The various numbers of $n$ are: $n \in \{1, 10, 20, 30, 40, 50, 100, 150, 200\}$. Note that $n=200$ corresponds to the MC estimate for both CartPole and MountainCar (as both environments terminate when reaching $t = 200$). 

As learning rate for the regular Advantage Estimation we use $lr_A \in \{0.001, 0.003, 0.005, 0.007, 0.009, 0.01\}$.
For the Generalized Advantage Estimation we use  $lr_{GAE} \in \{0.01, 0.03, 0.05, 0.07\}$. 

GAE uses a larger learning rate, because it reduces the variance more than the Advantage Estimation. This makes it able to use larger updates than the normal Advantage Estimation. 

The search for these optimal parameters is cubic, as we iterate over 3 separate parameters, namely $n$, $lr$ and PyTorch random seeds. To speed up the experiment we use multi-threaded agents where 16 environments were played at the same time, all with their own seeds. Additionally, multi-threading leads to better estimates of the gradient and reduces variance, allowing for a higher learning rate and improving convergence. 

### Reproducibility

To ensure reproducibility, we have manually set the seeds for PyTorch and the gym environments. 
We use in total 5 seeds, namely $[0, 30, 60, 90, 120]$, for _PyTorch_ and *NumPy*. The environments use seeds 16 to 31, where the seeds are determined by the number of workers (environment instances) we have. In our case this is 16, the seeds are calculated as followed:

```seeds = [i + num_envs for i in range(num_envs)]```

## Results and Analysis

To determine which setup works best, we first combine the results of all the seeds, sorted by return type, $n$ used in $n$-steps and learning rate. We then calculate the mean over the rewards, and use this to determine the best setup per return type. The results can be found in <a href="#best_lr">Table 1</a>.

|                                      | $n = 1$      | $n = 10$ | $n = 20$ | $n = 30$ | $n = 40$ | $n = 50$   | $n = 100$              | $n = 150$  | $n = 200$ |
| ------------------------------------ | ------------ | -------- | -------- | -------- | -------- | ---------- | ---------------------- | ---------- | --------- |
| **Generalized Advantage Estimation** | 0.03         | 0.01     | 0.01     | 0.01     | 0.03     | 0.01, 0.05 | 0.01, 0.03, 0.07, 0.09 | 0.03, 0.07 | 0.03      |
| **Advantage Estimation**             | 0.001, 0.003 | 0.001    | 0.01     | 0.009    | 0.005    | 0.007      | 0.007                  | 0.005      | 0.009     |

> *<span id="best_lr">Table 1</span>: Optimal learning rate per $n$-step. The value in each cell corresponds to learning rate which yielded the greatest average return. When there are multiple values present in a cell, the results were similar up to 1.0 difference in the return. An example for GAE, $n=150$; 0.03 yields a return of $180.7$ whereas 0.05 yields a return of $181.8$*.



For Generalized Advantage Estimation we see that, around $n = 100 $, there is an optimum in the amount of learning rates that lead to the optimal returns. For this $n$, many different learning rates lead to a good performance. A reason for this could be that the bias-variance trade-off is balanced here. In the next figures we show the return for AE and GAE of the best learning rates per $n$-step.



![Average Returns for different num steps](avg_return.png)
> *<span id="avg_returns">Figure 1</span>: These results are for the CartPole-v0 environment. We show results for the best learning rate of the GAE and AE returns. The graphs show the mean with surrounding it one standard deviation. The axis label "Number of steps (in thousands) refers to the steps taken in the environment themselves, and needs to also be multiplied by the number of agents. The y-axis is averaged over the seeds and the rewards observed at 1000-step interval. The returns are averaged by freezing the weights at each 1000th step and running an agent on 10 different episodes.*

The learning curves have been plotted in <a href="#avg_returns">Figure 1</a>, which show that GAE does not work for low values of $n$. We hypothesize that this is due to the bias that is added by GAE, whilst already being biased, which leads to a quick divergence. AE sometimes does manage to get high returns, because it is less biased, however displays higher variance. This especially becomes clear for $n=10$. Using standard AE shows a higher average return and less variance than using GAE. 

For $n>10$ using GAE already shows higher rewards and lower variance. It really starts to perform consistently good. This is not the case for regular AE, which seems to be harmed by larger values of $n$. 

For $n>50$, the return becomes more like Monte Carlo methods, we see that the learning curve for GAE is less steep i.e. more iterations are needed to reach optimal behaviour. This is due to the higher variance MC methods have. Also, it waits longer to backup, which slows the online learning down. 

For $n>100$ we see that both methods AE and GAE show lower returns and higher variance. Optimal is $n=100$ for which we see that the variance is really reduced to a minimum, even though high $n$-step/MC methods are inherently high variance. This shows that GAE is able to really reduce the variance, and is quite remarkable. 

## Conclusion

*What is the effect of (generalized) advantage estimation on the return in $n$-step bootstrapping?*

The idea of using GAE as a baseline in actor-critic methods is that is reduces variance while introducing a tolerable amount of bias compared to using the more standard AE. As $n$-step bootstrapping show high variance for higher $n$ and high bias for lower $n$ we hypothesized that GAE should work better for higher values of $n$.

What we see is that GAE does indeed outperform AE for higher value of $n$ as it shows much less variance. Also the learning curve for AE becomes gradually less steep when $n$-step approaches the MC method while this is not the case for GAE. However if $n$ is set too high, $n>100$, GAE will start to perform worse again, but will still be better than AE. 

Now it is important to keep in mind that the way these methods are tested is quite limited. In this experiment the returns showed are averaged over a total of 5 seeds, which could give misleading results. Also, the results obtained in this experiment could be specific for this environment. For further research we would suggest to test these methods on other environments to see if our conclusion generalizes well.





<link rel="stylesheet" href="https://unpkg.com/applause-button/dist/applause-button.css" />

<script src="https://unpkg.com/applause-button/dist/applause-button.js"></script>
<style>
  applause-button .count-container {
    margin:0 auto;
  }
  applause-button .count-container .count {
    text-size:1.2em;
  }
</style>
<div>
<applause-button url="antonsteenvoorden.nl" multiclap="true" style="margin: 0 auto; width: 58px; height: 58px;"/>
</div>


[^1]:  Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. *arXiv preprint arXiv:1506.02438*.
[^2]: https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
[^3]: https://github.com/ikostrikov/pytorch-a3c
[^4]: Actually, bootstrapping is what defines actor critic methods when contrasted to vanilla policy gradient methods.
[^5]: For the sake of completeness, we have briefly tested the model on a different environment, MountaintCar-v0, but we did not manage to converge for a selection of learning rates and $n$-steps in due time for this project.

