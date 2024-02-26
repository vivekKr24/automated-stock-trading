import numpy as np
import torch
import matplotlib.pyplot as plt
from train_utils import Agent, ActorCriticTrainer
from market_simulator.environment import Environment


class TradingTester:
    def __init__(self, env: Environment, trainer: ActorCriticTrainer):
        self.env = env
        self.trainer = trainer

        # Initialize parameters for plot_results
        self.action_list = []
        self.q_1_value_list = []
        self.q_2_value_list = []
        self.return_list = []
        self.holdings_list = []
        self.balance_list = []
        self.close_prices = []
        self.game_index = 0
        self.step_index = 0

    def test_agent(self, game_index, step_index):
        total_returns = 0
        current_state = self.env.initial_state()
        q_1_value_list = []
        q_2_value_list = []

        return_list = []
        holdings_list = []
        balance_list = []
        action_list = []
        close_prices = []
        message = f"Trading day index: {current_state.vector_index}"
        while not current_state:
            message = f"Trading day index: {current_state.vector_index}"
            print('\b' * len(message) + message, end='')
            agent_action = self.trainer.agent().action(current_state)

            state_action_pair = torch.cat([current_state().view(1, -1).cpu(), agent_action.view(1, -1).cpu()],
                                          dim=1).cuda()
            q_1 = self.trainer.critic_1_network(state_action_pair)
            q_2 = self.trainer.critic_2_network(state_action_pair)

            r, s_ = self.env.get_reward_and_next_state(current_state.cpu(), agent_action.cpu())
            total_returns += r

            q_1_value_list.append(q_1.clone().detach().item())
            q_2_value_list.append(q_2.clone().detach().item())
            action_list.append(agent_action.clone().detach().item())
            return_list.append(total_returns)
            holdings_list.append(current_state.portfolio.holdings[0] * 50)
            balance_list.append(current_state.portfolio.balance)
            close_prices.append(100 * current_state.portfolio.close_prices[0])

            current_state = s_

        print(f"\nTOTAL RETURNS: {total_returns} | FINAL PORTFOLIO VALUE: {current_state.portfolio.value()}")

        if game_index > 0:
            self.plot_results(
                action_list,
                q_1_value_list,
                q_2_value_list,
                return_list,
                holdings_list,
                balance_list,
                close_prices
            )

        # Update instance variables
        self.action_list = action_list
        self.q_1_value_list = q_1_value_list
        self.q_2_value_list = q_2_value_list
        self.return_list = return_list
        self.holdings_list = holdings_list
        self.balance_list = balance_list
        self.close_prices = close_prices
        self.game_index = game_index
        self.step_index = step_index

    def plot_results(self, action_list, q_1_value_list, q_2_value_list, return_list, holdings_list, balance_list,
                     close_prices):
        # Create a figure
        plt.figure(figsize=(8, 6))

        # Plot data on the first subplot
        plt.subplot(2, 1, 1)
        plt.plot(action_list, label='action')
        plt.plot(q_1_value_list, label='q_1')
        plt.plot(q_2_value_list, label='q_2')
        plt.grid(color='green', linestyle='--', linewidth=0.5)
        plt.legend()

        # Plot data on the second subplot
        plt.subplot(2, 1, 2)
        plt.plot(return_list)
        plt.plot(holdings_list)
        plt.plot(balance_list)
        plt.plot(close_prices)
        plt.grid(color='green', linestyle='--', linewidth=0.5)

        plt.legend(["RETURNS", "HOLDINGS", "BALANCE", "CLOSE"])

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig(f'TESTS/reports/game-summary/{self.game_index + 1}_{self.step_index}_portfolio.jpeg')
        # Show the plots

        self.plot_arrays(self.action_list, action_list)
        plt.close()

    def plot_arrays(self, array1, array2):
        delta = np.array(array2) - np.array(array1)

        # Create a figure
        plt.figure(figsize=(8, 6))

        # Plot array1
        plt.plot(array1)

        # Plot array2
        plt.plot(array2)

        # Highlight positive differences in green
        plt.fill_between(range(len(delta)), array1, array2, where=delta > 0, facecolor='green', interpolate=True, alpha=1)

        # Highlight negative differences in red
        plt.fill_between(range(len(delta)), array1, array2, where=delta < 0, facecolor='red', interpolate=True, alpha=1)

        # Customize the plot
        plt.title('Changes in action from prev episode to next episode')
        plt.legend(['PREV', 'CURR'])
        plt.grid(True)
        plt.savefig(f'TESTS/reports/game-summary/{self.game_index}_{self.step_index}_action_delta.jpeg')
        plt.close()

        # Show the plot

# Example Usage:
# Initialize your Environment, Agent, and Trainer
# env = Environment(...)
# agent = Agent(...)
# trainer = ActorCriticTrainer(...)

# Create an instance of TradingTester
# trading_tester = TradingTester(env, agent, trainer)

# Test the agent
# trading_tester.test_agent(game_index=1, step_index=1)
