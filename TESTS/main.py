import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from market_simulator.environment import Environment
from market_simulator.portfolio import Portfolio
from dataset.indicators import get_default_indicators
from metrics import TradingTester
from models.ciritic_net import CriticNet
from train_utils import ActorCriticTrainer
from dataset.dataset import TransitionDataset
from models.actor_net import ActorNet
import warnings

warnings.filterwarnings("ignore")

folder_path = "TESTS/reports/game-summary"
files = glob.glob(os.path.join(folder_path, '*'))
for file in files:
    try:
        os.remove(file)
        print(f"Deleted: {file}")
    except Exception as e:
        print(f"Error deleting {file}: {e}")


def should_update_networks(episode_idx, step_idx):
    return episode_idx > 5 and step_idx > 0
    # return True


torch.set_default_dtype(torch.float32)
print('Torch cuda:', torch.cuda.is_available())
torch.set_default_device('cuda')
tickers = ['HDFCBANK.NS']
indicators = get_default_indicators()
initial_holdings = np.zeros(shape=[len(tickers)])
initial_balance = 10000
portfolio = Portfolio(tickers=tickers, holdings=initial_holdings, balance=initial_balance)

environment = Environment(indicators=indicators, portfolio=portfolio)
state_size = environment.state_size

assert state_size in environment.initial_state()().size()
action_size = environment.action_size

actor_net = ActorNet(state_size=state_size, hidden_size=state_size - 7, action_size=action_size)
critic_1_net = CriticNet(state_size=state_size, hidden_size=state_size - 5, action_size=action_size)
critic_2_net = CriticNet(state_size=state_size, hidden_size=state_size - 5, action_size=action_size)

optimizer = torch.optim.Adagrad
trainer = ActorCriticTrainer(environment=environment,
                             actor_network=actor_net,
                             critic_1_network=critic_1_net,
                             critic_2_network=critic_2_net,
                             optimizer=optimizer)

n_episodes = 1000
n_steps = 1000
n_days = 5
n_actions = 10
trading_tester = TradingTester(environment, trainer)
trading_tester.test_agent(0, 0)
for episode in range(1, n_episodes + 1):
    current_state = environment.initial_state()
    print(f'Episode: {episode}')
    for step in tqdm(range(1, n_steps + 1)):
        transitions = TransitionDataset(capacity=100000)
        temp_state = current_state
        start_day = current_state.vector_index
        final_day = -1

        # print("Generating samples")
        for day in range(n_days):
            if bool(temp_state):
                break
            next_state = None
            for action_idx in range(n_actions):
                action = actor_net(temp_state().view(1, -1)).view(-1)
                action_noise = torch.randn_like(action)
                action = (action + action_noise)
                if temp_state.portfolio.balance < 100:
                    action = - torch.abs(action)

                if temp_state.vector_index == 45:
                    action = torch.abs(action)

                action = action * 10
                reward, next_state = environment.get_reward_and_next_state(temp_state, action)
                final_day = current_state.vector_index

                done = bool(next_state)

                transitions.add_transition(temp_state, action, next_state, reward, done)

            if bool(next_state):
                temp_state = current_state

            if day == n_days - 1:
                current_state = next_state

            temp_state = next_state

        # print(f"Played from {start_day} to {final_day}")

        if True:
            transition_loader = DataLoader(dataset=transitions, batch_size=64,
                                           generator=torch.Generator(device='cuda'))
            critic_1_loss = None
            critic_2_loss = None
            policy_loss = None
            for batch in transition_loader:
                policy_update = step % 20 == 0
                all_loss = trainer.update(batch=batch, policy_update=policy_update, update_target=policy_update)

                if critic_1_loss is None:
                    critic_1_loss = all_loss[0]
                else:
                    critic_1_loss = critic_1_loss + all_loss[0]

                if critic_2_loss is None:
                    critic_2_loss = all_loss[1]
                else:
                    critic_2_loss = critic_2_loss + all_loss[1]

                if len(all_loss) == 3:
                    if policy_loss is None:
                        policy_loss = all_loss[2]
                    else:
                        policy_loss = policy_loss + all_loss[2]

            if policy_loss is not None:
                policy_loss.backward(retain_graph=True)
                trainer.critic_1_optimizer.zero_grad()
                # print('Policy Updated')
            critic_1_loss.backward(retain_graph=True)
            critic_2_loss.backward(retain_graph=True)

            trainer.step()
            if step == n_steps - 1:
                print(f'STEP: {step} | '
                      f'CRITIC_1_LOSS: {critic_1_loss.cpu().detach().item():.2f} |'
                      f' CRITIC_2_LOSS: {critic_2_loss.cpu().detach().item():.2f} | ')
                      # f'POLICY LOSS {policy_loss.cpu().detach().item():.2f}')

    trading_tester.test_agent(game_index=episode, step_index=0)
