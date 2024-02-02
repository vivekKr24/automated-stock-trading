import torch

from market_simulator.portfolio import Portfolio


class State:
    def __init__(self, environment, vector, vector_index, portfolio):
        self.environment = environment
        self.vector_index = vector_index
        self.portfolio: Portfolio = portfolio
        self.vector = torch.tensor([*portfolio, *vector], dtype=torch.float32)

    def __call__(self):
        return self.vector

    def __bool__(self):
        return not (self.vector_index + 1 < self.environment.state_data.shape[0])

    def __str__(self):
        return f'{self.vector.numpy()}'

    def cpu(self):
        return State(environment=self.environment, vector=self.vector.cpu(), vector_index=self.vector_index, portfolio=self.portfolio)
