from typing import Dict

from dataset.data_source import DataSource
from market_simulator.portfolio import Portfolio
from market_simulator.state import State


class Environment:
    def __init__(self, indicators, portfolio: Portfolio):
        self.indicators = indicators
        self.portfolio = portfolio

        data_src = DataSource(indicators, portfolio.tickers)

        self.state_data = data_src.load_dataset()

        # Technical indicators and close price
        self.ticker_data_len = len(self.indicators.keys()) + 1
        self.action_size = len(portfolio.tickers)

        # [indicators, close, holding] for every ticker, and balance
        self.state_size = self.action_size * (self.ticker_data_len + 1) + 1

    def indices(self, attr):
        if attr == 'Close' or attr == 0:
            return [0 + i * self.ticker_data_len for i in range(self.action_size)]

    def get_reward_and_next_state(self, current_state: State, action):
        next_state_index = current_state.vector_index + 1
        next_day_close_prices = self.state_data[next_state_index, self.indices('Close')]
        portfolio = current_state.portfolio.update(action, next_day_close_prices)
        next_state = State(self, self.state_data[next_state_index], next_state_index, portfolio)
        reward = next_state.portfolio.value() - current_state.portfolio.value()
        return reward, next_state

    def initial_state(self, start_index=45) -> State:
        self.portfolio.close_prices = self.state_data[start_index, self.indices('Close')]
        return State(self, self.state_data[start_index], start_index, self.portfolio)
