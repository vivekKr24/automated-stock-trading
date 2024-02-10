import numpy as np
import torch


class Portfolio:
    def __init__(self, tickers, holdings, balance, closing_prices=None):
        self.tickers = tickers
        self.holdings = holdings
        self.balance = balance
        self.close_prices = closing_prices
        self.risk_free_rate = 0
        # self.risk_free_rate = 0.08 / 260

    def __iter__(self):
        return iter([self.balance, *(self.holdings.tolist())])

    def update(self, action, next_day_close_prices):
        action_no_grad = action.detach().cpu().numpy()
        sell_actions = -np.clip(action_no_grad, -np.inf, 0)
        buy_actions = np.clip(action_no_grad, 0, np.inf)

        # Process sell actions
        effective_sell_actions = np.minimum(sell_actions, self.holdings)
        sell_gain = np.sum(effective_sell_actions * self.close_prices)
        holdings = self.holdings - effective_sell_actions
        balance = self.balance * (1 + self.risk_free_rate) + sell_gain

        # Process buy actions
        cash_reqs = np.multiply(buy_actions, self.close_prices)
        total_cash_reqs = np.sum(cash_reqs)
        if 0 < total_cash_reqs <= self.balance:
            effective_buy_action = cash_reqs / total_cash_reqs
            holdings = self.holdings + effective_buy_action
            balance = balance - np.sum(effective_buy_action * self.close_prices)

        return Portfolio(self.tickers, holdings, balance, next_day_close_prices)

    def value(self):
        if self.close_prices is None:
            raise Exception("Portfolio not associated with an environment, create an environment first")
        stock_value = self.close_prices * self.holdings
        return np.sum(stock_value) + self.balance
