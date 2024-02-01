import torch
from torch.utils.data import DataLoader

from market_simulator.environment import Environment
from market_simulator.portfolio import Portfolio
from dataset.indicators import get_default_indicators
from models.ciritic_net import CriticNet
from train_utils import ActorCriticTrainer
import tqdm

torch.set_default_dtype(torch.float32)

tickers = ['MSFT']
indicators = get_default_indicators()
initial_holdings = torch.zeros(size=[len(tickers)])
initial_balance = 1000
portfolio = Portfolio(tickers=tickers, holdings=initial_holdings, balance=initial_balance)
portfolio

environment = Environment(indicators=indicators, portfolio=portfolio)
state_size = environment.state_size
action_size = environment.action_size
environment, state_size, action_size

from models.actor_net import ActorNet

actor_net = ActorNet(state_size=state_size, hidden_size=state_size - 5, action_size=action_size)
critic_1_net = CriticNet(state_size=state_size, hidden_size=state_size - 5, action_size=action_size)
critic_2_net = CriticNet(state_size=state_size, hidden_size=state_size - 5, action_size=action_size)

optimizer = torch.optim.Adam
trainer = ActorCriticTrainer(environment=environment,
                             actor_network=actor_net,
                             critic_1_network=critic_1_net,
                             critic_2_network=critic_2_net,
                             optimizer=optimizer)


def should_update_networks(episode_idx, step_idx):
    return episode_idx > 15 and step_idx > 0


from dataset.dataset import TransitionDataset

n_episodes = 1000
n_steps = 1000

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
            transition_loader = DataLoader(dataset=transitions, batch_size=32, shuffle=True, drop_last=True)
            for batch in tqdm.tqdm(transition_loader):
                all_loss = trainer.update(batch=batch, policy_update=(step % 10 == 0))

    print(episode)

