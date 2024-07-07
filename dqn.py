import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import namedtuple, deque
from typing import Tuple, List, Any

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, h: int, w: int, outputs: int) -> None:
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)

        def conv2d_size_out(size: int, kernel_size: int, stride: int) -> int:
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(size=conv2d_size_out(size=conv2d_size_out(size=w, kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3, stride=1)
        convh = conv2d_size_out(size=conv2d_size_out(size=conv2d_size_out(size=h, kernel_size=8, stride=4), kernel_size=4, stride=2), kernel_size=3, stride=1)
        linear_input_size = convw * convh * 64
        self.head = nn.Linear(in_features=linear_input_size, out_features=outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args: Any) -> None:
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)