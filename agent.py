import math
import random
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from itertools import count
import matplotlib.pyplot as plt
from dqn import DQN, ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
MEMORY_SIZE = 10000

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.n_actions = env.action_space.n
        self.init_screen = env.reset()[0]
        self.screen_height, self.screen_width = self.preprocess_frame(frame=self.init_screen).shape
        
        self.policy_net = DQN(h=self.screen_height, w=self.screen_width, outputs=self.n_actions).to(device=device)
        self.target_net = DQN(h=self.screen_height, w=self.screen_width, outputs=self.n_actions).to(device=device)
        self.target_net.load_state_dict(state_dict=self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(params=self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(capacity=MEMORY_SIZE)
        
        self.steps_done = 0
        self.episode_durations = []

    def preprocess_frame(self, frame):
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(src=frame, dsize=(84, 84))
        frame = frame / 255.0
        return frame

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(dim=1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(batch_size=BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(data=tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)

        next_state_values = torch.zeros(size=(BATCH_SIZE,), device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(dim=1)[0]
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(input=state_action_values, target=expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(parameters=self.policy_net.parameters(), clip_value=100)
        self.optimizer.step()

    def update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(state_dict=target_net_state_dict)

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(data=self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(dimension=0, size=100, step=1).mean(dim=1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(interval=0.001)

    def train(self, num_episodes):
        for i_episode in range(num_episodes):
            state, _ = self.env.reset()
            state = self.preprocess_frame(frame=state)
            state = np.stack([state] * 4, axis=0)  # Stack 4 frames for the initial state
            state = torch.from_numpy(state).unsqueeze(dim=0).to(device=device, dtype=torch.float32)

            for t in count():
                action = self.select_action(state=state)
                observation, reward, terminated, truncated, _ = self.env.step(action=action.item())
                reward = torch.tensor(data=[reward], device=device)
                done = terminated or truncated

                if not done:
                    next_frame = self.preprocess_frame(frame=observation)
                    next_state = np.append(state.cpu().numpy()[0, 1:], np.expand_dims(next_frame, axis=0), axis=0)
                    next_state = torch.from_numpy(next_state).unsqueeze(dim=0).to(device=device, dtype=torch.float32)
                else:
                    next_state = None

                self.memory.push(state, action, next_state, reward)
                state = next_state
                self.optimize_model()

                self.update_target_network()

                if done:
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break

        print('Complete')
        self.plot_durations(show_result=True)
        plt.ioff()
        plt.show()