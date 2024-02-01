import numpy as np
import torch


class Portfolio:
    def __init__(self, tickers, holdings, balance, closing_prices=None):
        self.tickers = tickers
        self.holdings = holdings
        self.balance = torch.tensor(balance, dtype=torch.float32)
        self.close_prices = closing_prices
        self.risk_free_rate = 0.08 / 260

    def __iter__(self):
        return iter([self.balance, *(self.holdings.numpy().tolist())])

    def update(self, action, next_day_close_prices):
        action_no_grad = action.detach()
        sell_actions = -torch.clip(action_no_grad, -torch.inf, 0)
        buy_actions = torch.clip(action_no_grad, 0, torch.inf)

        # Process sell actions
        effective_sell_actions = torch.minimum(sell_actions, self.holdings)
        sell_gain = torch.sum(effective_sell_actions * torch.tensor(self.close_prices))
        holdings = self.holdings - effective_sell_actions
        balance = self.balance * (1 + self.risk_free_rate) + sell_gain

        # Process buy actions
        cash_reqs = torch.multiply(buy_actions, torch.tensor(self.close_prices))
        total_cash_reqs = torch.sum(cash_reqs)
        if cash_reqs > 0 and self.balance >= total_cash_reqs:
            effective_buy_action = cash_reqs / total_cash_reqs
            holdings = self.holdings + effective_buy_action
            balance = balance - torch.sum(effective_buy_action * torch.tensor(self.close_prices))

        return Portfolio(self.tickers, holdings, balance, next_day_close_prices)

    def value(self):
        if self.close_prices is None:
            raise Exception("Portfolio not associated with an environment, create an environment first")
        stock_value = torch.tensor(self.close_prices, dtype=torch.float32) * self.holdings
        return torch.sum(stock_value) + self.balance
