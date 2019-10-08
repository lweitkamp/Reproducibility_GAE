import torch.nn as nn


class ActorCriticMLP(nn.Module):
    def __init__(self, input_dim, n_acts, n_hidden=32):
        super(ActorCriticMLP, self).__init__()
        self.input_dim = input_dim
        self.n_acts = n_acts
        self.n_hidden = n_hidden

        self.features = nn.Sequential(
            nn.Linear(self.input_dim, self.n_hidden),
            nn.ReLU(),
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

