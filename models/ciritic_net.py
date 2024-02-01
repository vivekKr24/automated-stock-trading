import torch.nn as nn


class CriticNet(nn.Module):
    def __init__(self, state_size, hidden_size, action_size, soft_update_rate=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_update_rate = soft_update_rate

        # Policy Value Layers
        self.critic_input_layer = nn.Linear(state_size + action_size, hidden_size)
        self.critic_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.critic_layer_3 = nn.Linear(hidden_size, 1)

        # Policy Target Layers
        self.critic_input_layer_target = nn.Linear(state_size + action_size, hidden_size)
        self.critic_layer_2_target = nn.Linear(hidden_size, hidden_size)
        self.critic_layer_3_target = nn.Linear(hidden_size, 1)

        self.critic_activation_fn = nn.ReLU()

    def forward(self, x):
        x = self.critic_activation_fn(self.critic_input_layer(x.float()))
        x = self.critic_activation_fn(self.critic_layer_2(x))
        x = self.critic_layer_3(x)

        return x

    def target(self, x):
        x = self.critic_activation_fn(self.critic_input_layer_target(x.float()))
        x = self.critic_activation_fn(self.critic_layer_2_target(x))
        x = self.critic_layer_3_target(x)

        return x.detach()

    def update_target_network(self):
        targ_param_index = len(list(self.parameters())) / 2
        for param, target_param in zip(self.parameter()[:targ_param_index], self.parameter()[targ_param_index:]):
            soft_update = self.soft_update_rate * param + (1 - self.soft_update_rate) * target_param
            target_param.data.copy_(soft_update)
