import random

import torch
from torch.utils.data import DataLoader

from market_simulator.environment import Environment
from market_simulator.portfolio import Portfolio
from dataset.indicators import get_default_indicators
from metrics import test_agent
from models.ciritic_net import CriticNet
from train_utils import ActorCriticTrainer
import tqdm

from dataset.dataset import TransitionDataset

torch.set_default_dtype(torch.float32)
print('Torch cuda:', torch.cuda.is_available())
torch.set_default_device('cuda')
tickers = ['MSFT']
indicators = get_default_indicators()
initial_holdings = torch.zeros(size=[len(tickers)])
initial_balance = 1000
portfolio = Portfolio(tickers=tickers, holdings=initial_holdings, balance=initial_balance)

environment = Environment(indicators=indicators, portfolio=portfolio)
state_size = environment.state_size

assert state_size in environment.initial_state()().size()
action_size = environment.action_size

from models.actor_net import ActorNet

actor_net = ActorNet(state_size=state_size, hidden_size=state_size - 7, action_size=action_size)
critic_1_net = CriticNet(state_size=state_size, hidden_size=state_size - 5, action_size=action_size)
critic_2_net = CriticNet(state_size=state_size, hidden_size=state_size - 5, action_size=action_size)

optimizer = torch.optim.Adam
trainer = ActorCriticTrainer(environment=environment,
                             actor_network=actor_net,
                             critic_1_network=critic_1_net,
                             critic_2_network=critic_2_net,
                             optimizer=optimizer)


def should_update_networks(episode_idx, step_idx):
    return episode_idx > 20 and step_idx > 0


n_episodes = 1000
n_steps = 250

transitions = TransitionDataset(capacity=100000)
for episode in range(1, n_episodes + 1):

    current_state = environment.initial_state()
    for step in range(1, n_steps + 1):
        action = actor_net(current_state())
        action_noise = torch.randn_like(action)
        reward, next_state = environment.get_reward_and_next_state(current_state, action)

        done = bool(next_state)

        if done:
            break

        transitions.add_transition(current_state, action, next_state, reward, done)

        if should_update_networks(episode, step):
            random_indices = random.sample(range(len(transitions)), 1000)
            random_batch = [transitions[i] for i in random_indices]
            transition_loader = DataLoader(dataset=random_batch, batch_size=64, drop_last=True,
                                           generator=torch.Generator(device='cuda'))
            critic_1_loss = torch.tensor(0., dtype=torch.float32)
            critic_2_loss = torch.tensor(0., dtype=torch.float32)
            policy_loss = torch.tensor(0., dtype=torch.float32)
            for batch in transition_loader:
                all_loss = trainer.update(batch=batch, policy_update=bool(step % 10 == 0))
                critic_1_loss = critic_1_loss + all_loss[0]
                critic_2_loss = critic_2_loss + all_loss[1]
                if len(all_loss) == 3:
                    policy_loss = policy_loss + all_loss[2]

            if policy_loss.grad is not None:
                policy_loss.backward(retain_graph=True)
            critic_1_loss.backward(retain_graph=True)
            critic_2_loss.backward(retain_graph=True)

            trainer.step()

            print(f'EPISODE: {episode:6d} | STEP: {step:6d} | '
                  f'CRITIC_1_LOSS: {critic_1_loss.detach().item():.2f} |'
                  f' CRITIC_2_LOSS: {critic_2_loss.detach().item():.2f} | '
                  f'POLICY_LOSS: {policy_loss.detach().item():.2f}')

            if step % 10:
                test_agent(environment, trainer.agent(), episode)
