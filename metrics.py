import torch
import matplotlib.pyplot as plt
from train_utils import Agent, ActorCriticTrainer
from market_simulator.environment import Environment


def test_agent(_env: Environment, _agent: Agent, game_index, step_index, trainer: ActorCriticTrainer):
    total_returns = 0
    _current_state = _env.initial_state()
    q_1_value_list = []
    q_2_value_list = []

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

        state_action_pair = torch.cat([_current_state().view(1, -1).cpu(), agent_action.view(1, -1).cpu()], dim=1).cuda()
        q_1 = trainer.critic_1_network(state_action_pair)
        q_2 = trainer.critic_2_network(state_action_pair)

        q_1_value_list.append(q_1.clone().detach().item())
        q_2_value_list.append(q_2.clone().detach().item())

        action_list.append(agent_action.clone().detach().item())

        r, s_ = _env.get_reward_and_next_state(_current_state.cpu(), agent_action.cpu())
        total_returns += r
        return_list.append(total_returns)
        holdings_list.append(_current_state.portfolio.holdings[0] * 50)
        balance_list.append(_current_state.portfolio.balance)
        close_prices.append(_current_state.portfolio.close_prices[0])

        _current_state = s_

    print(f"\nTOTAL RETURNS: {total_returns} | FINAL PORTFOLIO VALUE: {_current_state.portfolio.value()}")
    # Create a figure
    plt.figure(figsize=(8, 6))

    # Plot data on the first subplot
    plt.subplot(2, 1, 1)
    plt.plot(action_list, label='action')
    plt.plot(q_1_value_list, label='q_1')
    plt.plot(q_1_value_list, label='q_2')
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
    plt.savefig(f'TESTS/reports/game-summary/{game_index}_{step_index}.jpeg')
    plt.close()
    # Show the plots
    plt.show()

