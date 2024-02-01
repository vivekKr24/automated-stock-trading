import torch
from torch.utils.data import Dataset
from torchviz import make_dot

from models.actor_net import ActorNet
from models.ciritic_net import CriticNet


class ActorCriticTrainer:
    def __init__(self, environment, actor_network: ActorNet, critic_1_network: CriticNet, critic_2_network: CriticNet,
                 optimizer, policy_update_delay=10
                 ):
        self.actor_network = actor_network
        self.critic_1_network = critic_1_network
        self.critic_2_network = critic_2_network
        self.optimizer = optimizer

        self.actor_optimizer = optimizer(actor_network.parameters())
        self.critic_1_optimizer = optimizer(critic_1_network.parameters())
        self.critic_2_optimizer = optimizer(critic_2_network.parameters())

    def update(self, batch, policy_update=False):
        actor_optimizer = self.actor_optimizer
        critic_1_optimizer = self.critic_1_optimizer
        critic_2_optimizer = self.critic_2_optimizer
        actor_net = self.actor_network
        critic_1_net = self.critic_1_network
        critic_2_net = self.critic_2_network

        states, actions, rewards, next_states, dones = batch


        current_state_action_pairs = torch.cat((states, actions), dim=1)
        next_state_actions = actor_net.target(next_states)
        next_state_action_pairs = torch.cat((next_states, next_state_actions), dim=1)
        target_q_value1 = critic_1_net.target(next_state_action_pairs)
        target_q_value2 = critic_2_net.target(next_state_action_pairs)

        q_values = critic_1_net(current_state_action_pairs)
        losses = []

        if policy_update:
            policy_loss = - torch.mean(q_values)
            policy_loss.backward()
            make_dot(policy_loss, params=actor_net.named_parameters()).render('policy_loss')
            losses.append(policy_loss)
            actor_optimizer.step()

        target_q = torch.min(target_q_value1, target_q_value2)
        ones = torch.ones_like(dones).float()
        done_factor = (ones - dones).view(ones.size()[0], 1)
        y = rewards.view(rewards.size()[0], 1) + done_factor * target_q
        y = y.detach().requires_grad_(True)

        critic_1_input = torch.cat((states, actions.detach()), dim=1)
        critic_1_output = critic_1_net(critic_1_input)
        loss_fn = torch.nn.MSELoss()
        critic_1_loss = loss_fn(critic_1_output, y).view(1)

        critic_2_input = torch.cat((states, actions.detach()), dim=1)
        critic_2_output = critic_2_net(critic_2_input)
        critic_2_loss = loss_fn(critic_2_output, y).view(1)

        make_dot(critic_1_loss, params=dict(critic_1_net.named_parameters())).render('critic_1_loss')
        make_dot(critic_2_loss, params=dict(critic_2_net.named_parameters())).render('critic_2_loss')

        critic_1_loss.backward()
        critic_2_loss.backward()

        critic_1_optimizer.step()
        critic_2_optimizer.step()

        losses = [*losses, critic_1_loss, critic_2_loss]


        return losses
