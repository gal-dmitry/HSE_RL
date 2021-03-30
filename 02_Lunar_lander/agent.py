import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")

    def act(self, state):
        state = torch.tensor(state).cuda()
        state = state.float().unsqueeze(0)
        action = self.model(state)[0].max(0)[1].view(1, 1).item()
        return action

    def reset(self):
        pass