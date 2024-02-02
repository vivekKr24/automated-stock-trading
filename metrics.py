import torch
import matplotlib.pyplot as plt
from train_utils import Agent
from market_simulator.environment import Environment


def test_agent(_env: Environment, _agent: Agent, game_index):
    total_returns = 0
    _current_state = _env.initial_state()

    return_list = []
    holdings_list = []
    balance_list = []
    action_list = []
    close_prices = []

    message = f"Trading day index: {_current_state.vector_index}"
    while not _current_state:
        message = f"Trading day index: {_current_state.vector_index}"
        print('\b' * len(message) + message, end='')
        agent_action = _agent.action(_current_state)

        action_list.append(agent_action.clone().detach().item())

        r, s_ = _env.get_reward_and_next_state(_current_state.cpu(), agent_action.cpu())
        total_returns += r
        return_list.append(total_returns)
        holdings_list.append(_current_state.portfolio.holdings.cpu()[0] * 50)
        balance_list.append(_current_state.portfolio.balance.cpu())
        close_prices.append(_current_state.portfolio.close_prices.cpu()[0])

        assert total_returns != torch.nan

        _current_state = s_

    print(f"\nTOTAL RETURNS: {total_returns} | FINAL PORTFOLIO VALUE: {_current_state.portfolio.value()}")
    # Create a figure
    plt.figure(figsize=(8, 6))

    # Plot data on the first subplot
    plt.subplot(2, 1, 1)
    plt.plot(action_list, label='action')
    plt.title('Action')
    plt.legend()

    # Plot data on the second subplot
    plt.subplot(2, 1, 2)
    plt.plot(return_list)
    plt.plot(holdings_list)
    plt.plot(balance_list)
    plt.plot(close_prices)

    plt.legend(["RETURNS", "HOLDINGS", "BALANCE", "CLOSE"], loc="lower right")

    # plt.show()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f'metrics/game_{game_index}.jpeg')
    plt.close()
    # Show the plots
    plt.show()

