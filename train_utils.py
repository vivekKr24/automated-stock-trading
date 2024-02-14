import torch
import copy

from torchviz import make_dot

from market_simulator.state import State
from models.actor_net import ActorNet
from models.ciritic_net import CriticNet


def actor_param_hook(grad):
    print(f"POLICY GRADIENT: {grad.cpu().detach().numpy()}")


class ActorCriticTrainer:
    def __init__(self, environment, actor_network: ActorNet, critic_1_network: CriticNet, critic_2_network: CriticNet,
                 optimizer, policy_update_delay=10
                 ):
        self.actor_network = actor_network
        self.critic_1_network = critic_1_network
        self.critic_2_network = critic_2_network
        self.optimizer = optimizer

        # for name, param in dict(self.actor_network.named_parameters()).items():
        #     _name = name + ''
        #     param.register_hook(lambda grad: print(f"{_name} gradient: {grad.data.cpu().detach().numpy()}"))
        #     print(f"Hook added to {name}")

        # for name, param in dict(self.critic_1_network.named_parameters()).items():
        #     param.register_hook(lambda grad: print(f"{name} gradient: {grad.data.cpu().detach().numpy()}"))
        #     print(f"Hook added to {name}")
        #
        # for name, param in dict(self.critic_2_network.named_parameters()).items():
        #     param.register_hook(lambda grad: print(f"{name} gradient: {grad.data.cpu().detach().numpy()}"))
        #     print(f"Hook added to {name}")

        self.actor_optimizer = optimizer(actor_network.parameters(), lr=5e-2)
        self.critic_1_optimizer = optimizer(critic_1_network.parameters(), lr=1e-2)
        self.critic_2_optimizer = optimizer(critic_2_network.parameters(), lr=1e-2)

        self.tau = 0.5

    def step(self):
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()

        self.critic_1_optimizer.step()
        self.critic_1_optimizer.zero_grad()

        self.critic_2_optimizer.step()
        self.critic_2_optimizer.zero_grad()

    def update(self, batch, policy_update=False, update_target=False):
        actor_net = self.actor_network
        critic_1_net = self.critic_1_network
        critic_2_net = self.critic_2_network
        actor_net.train()
        critic_1_net.train()
        critic_2_net.train()

        states, actions, rewards, next_states, dones, current_state_action_pairs = batch
        rewards = rewards.float()
        next_state_actions = actor_net.target(next_states)
        next_state_action_pairs = torch.cat((next_states, next_state_actions), dim=1)
        target_q_value1 = critic_1_net.target(next_state_action_pairs)
        target_q_value2 = critic_2_net.target(next_state_action_pairs)

        losses = []

        target_q = torch.min(target_q_value1, target_q_value2)
        ones = torch.ones_like(dones).float()
        done_factor = (ones - dones).view(ones.size()[0], 1)
        y = 0.95 * done_factor * target_q + rewards.view(rewards.size()[0], 1)
        y = y.detach()

        states_no_grad = states.clone().detach()
        actions_no_grad = actions.clone().detach()

        critic_1_output = critic_1_net(torch.cat((states_no_grad, actions_no_grad), dim=1))
        loss_fn = torch.nn.MSELoss()
        critic_1_loss = loss_fn(critic_1_output, y)
        self.critic_1_optimizer.zero_grad()

        critic_2_output = critic_2_net(torch.cat((states_no_grad, actions_no_grad), dim=1))
        critic_2_loss = loss_fn(critic_2_output, y)
        self.critic_1_optimizer.zero_grad()

        # make_dot(critic_1_loss, params=dict(critic_1_net.named_parameters())).render('critic_1_loss')
        # make_dot(critic_2_loss, params=dict(critic_2_net.named_parameters())).render('critic_2_loss')

        if policy_update:
            q_values = critic_1_net(current_state_action_pairs)
            policy_loss = - torch.mean(q_values)
            # (make_dot(policy_loss,
            #           params=dict({**dict(actor_net.named_parameters()), **dict(critic_1_net.named_parameters())}))
            #  .render('policy_loss'))
            losses.append(policy_loss)
        if update_target:
            actor_net.soft_update_target_networks(self.tau)
            critic_1_net.soft_update_target_networks(self.tau)
            critic_2_net.soft_update_target_networks(self.tau)

        losses = [critic_1_loss, critic_2_loss, *losses]

        return losses

    def agent(self):
        return Agent(self)


class Agent:
    def __init__(self, trainer: ActorCriticTrainer):
        model_copy = copy.deepcopy(trainer.actor_network.cpu())
        trainer.actor_network.cuda()
        self.policy = model_copy

    def action(self, state: State, cpu=True):
        if cpu:
            return self.policy(state().cpu().view(1, -1)).view(-1)
        return self.policy(state().cpu().view(1, -1)).view(-1)
