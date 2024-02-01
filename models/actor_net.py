import torch
import torch.nn as nn


class ActorNet(nn.Module):
    def __init__(self, state_size, hidden_size, action_size, soft_update_rate=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_update_rate = soft_update_rate

        # Policy Value Layers
        self.actor_input_layer = nn.Linear(state_size, hidden_size)
        self.actor_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.actor_layer_3 = nn.Linear(hidden_size, action_size)

        # Policy Target Layers
        self.actor_input_layer_target = nn.Linear(state_size, hidden_size)
        self.actor_layer_2_target = nn.Linear(hidden_size, hidden_size)
        self.actor_layer_3_target = nn.Linear(hidden_size, action_size)

        self.actor_activation_fn = nn.Tanh()

    def forward(self, x):
        x = self.actor_activation_fn(self.actor_input_layer(x))
        x = self.actor_activation_fn(self.actor_layer_2(x))
        x = self.actor_layer_3(x)

        return x

    def target(self, x):
        x = self.actor_activation_fn(self.actor_input_layer_target(x))
        x = self.actor_activation_fn(self.actor_layer_2_target(x))
        x = self.actor_layer_3_target(x)

        return x.detach()

    def update_target_network(self):
        targ_param_index = len(list(self.parameters())) / 2
        for param, target_param in zip(self.parameter()[:targ_param_index], self.parameter()[targ_param_index:]):
            soft_update = self.soft_update_rate * param + (1 - self.soft_update_rate) * target_param
            target_param.data.copy_(soft_update)
