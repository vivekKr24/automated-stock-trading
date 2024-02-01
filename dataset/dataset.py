import os

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def __getitem__(self, idx):
        if type(idx) is str:
            index = ['state', 'action', 'reward', 'next_state', 'done']
            l = [x[index.index(idx)] for x in self.buffer]
            return torch.stack(l)
        return self.buffer[idx]

    def __len__(self):
        return len(self.buffer)

    def add_transition(self, state, action, next_state, reward, done):
        transition = (state(), action, reward, next_state(), torch.tensor(float(done)))

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            self.position = (self.position + 1) % self.capacity
