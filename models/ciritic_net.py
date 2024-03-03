from typing import Iterator, Tuple

import torch.nn as nn
from torch.nn import Parameter, init


class CriticNet(nn.Module):
    def __init__(self, state_size, hidden_size, action_size, soft_update_rate=0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soft_update_rate = soft_update_rate

        # Policy Value Layers
        self.critic_input_layer = nn.Linear(state_size + action_size, hidden_size)
        self.batch_norm_1 = nn.InstanceNorm1d(hidden_size)  # Add BatchNorm
        # self.critic_layer_2 = nn.Linear(hidden_size, hidden_size)
        # self.batch_norm_2 = nn.InstanceNorm1d(hidden_size)  # Add BatchNorm
        self.critic_layer_3 = nn.Linear(hidden_size, 1)

        # Policy Target Layers
        self.critic_input_layer_target = nn.Linear(state_size + action_size, hidden_size)
        self.batch_norm_1_target = nn.InstanceNorm1d(hidden_size)  # Add BatchNorm
        # self.critic_layer_2_target = nn.Linear(hidden_size, hidden_size)
        # self.batch_norm_2_target = nn.InstanceNorm1d(hidden_size)  # Add BatchNorm
        self.critic_layer_3_target = nn.Linear(hidden_size, 1)

        self.critic_activation_fn = nn.ReLU()

    def forward(self, x):
        x = self.critic_activation_fn(self.batch_norm_1(self.critic_input_layer(x.float())))
        # x = self.critic_activation_fn(self.batch_norm_2(self.critic_layer_2(x)))
        x = self.critic_layer_3(x)

        return x

    def parameters(self, recurse: bool = True):
        param_len = len(list(super().named_parameters())) // 2
        for name, param in list(self.named_parameters(recurse=recurse))[:param_len]:
            if name.count('target') == 0:
                yield param

    # def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True):
    #     param_len = len(list(self.named_parameters())) // 2
    #     for name, param in list(self.named_parameters(recurse=recurse))[:param_len]:
    #         if name.count('target') == 0:
    #             yield param

    def target(self, x):
        x = self.critic_activation_fn(self.batch_norm_1_target(self.critic_input_layer_target(x.float())))
        # x = self.critic_activation_fn(self.batch_norm_2_target(self.critic_layer_2_target(x)))
        x = self.critic_layer_3_target(x)

        return x.detach()

    def soft_update_target_networks(self, tau):
        self.tau = tau
        for target_param_name, param_name in zip(self.critic_input_layer_target._parameters.keys(),
                                                 self.critic_input_layer._parameters.keys()):
            target_param = getattr(self.critic_input_layer_target, target_param_name)
            param = getattr(self.critic_input_layer, param_name)
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        # for target_param_name, param_name in zip(self.critic_layer_2_target._parameters.keys(),
        #                                          self.critic_layer_2._parameters.keys()):
        #     target_param = getattr(self.critic_layer_2_target, target_param_name)
        #     param = getattr(self.critic_layer_2, param_name)
        #     target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)

        for target_param_name, param_name in zip(self.critic_layer_3_target._parameters.keys(),
                                                 self.critic_layer_3._parameters.keys()):
            target_param = getattr(self.critic_layer_3_target, target_param_name)
            param = getattr(self.critic_layer_3, param_name)
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
