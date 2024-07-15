import os

if os.path.basename(os.getcwd()) != 'video-pinball-rl':
    os.chdir("./video-pinball-rl")


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
import gymnasium as gym
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(self, env: gym.Env, batch_size: int, gamma: float, eps_start: float, eps_end: float, eps_decay: int, tau: float, lr: float, memory_size: int, path: str) -> None:
        self.env = env
        self.render_mode = env.render_mode
        self.n_actions = env.action_space.n
        self.init_screen = env.reset()[0]
        self.screen_height, self.screen_width = self.preprocess_frame(frame=self.init_screen).shape
        
        self.policy_net = DQN(h=self.screen_height, w=self.screen_width, outputs=self.n_actions).to(device=device)
        self.target_net = DQN(h=self.screen_height, w=self.screen_width, outputs=self.n_actions).to(device=device)
        self.target_net.load_state_dict(state_dict=self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.AdamW(params=self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(capacity=memory_size)
        
        self.steps_done = 0
        self.epoch_durations = []
        self.epoch_rewards = []
        self.losses = []
        self.epsilons = []

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.memory_size = memory_size
        self.path = path

        # Create directories if not exist
        os.makedirs(self.path, exist_ok=True)
        self.metrics_files = {
            'durations': f'{self.path}/epoch_durations.csv',
            'rewards': f'{self.path}/epoch_rewards.csv',
            'losses': f'{self.path}/losses.csv',
            'epsilons': f'{self.path}/epsilons.csv'
        }
        self.initialize_metrics_files()

    def initialize_metrics_files(self) -> None:
        for metric, filepath in self.metrics_files.items():
            if not os.path.exists(filepath):
                with open(filepath, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([metric])

    def save_metrics(self) -> None:
        with open(self.metrics_files['durations'], 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.epoch_durations[-1:])

        with open(self.metrics_files['rewards'], 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.epoch_rewards[-1:])

        with open(self.metrics_files['losses'], 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.losses[-1:])

        with open(self.metrics_files['epsilons'], 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.epsilons[-1:])

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        # Convert to grayscale and resize
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(src=frame, dsize=(84, 84))
        frame = frame / 255.0
        return frame

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        self.epsilons.append(eps_threshold)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(dim=1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def optimize_model(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(batch_size=self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(data=tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(dim=1, index=action_batch)

        next_state_values = torch.zeros(size=(self.batch_size,), device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(dim=1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(input=state_action_values, target=expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(parameters=self.policy_net.parameters(), clip_value=100)
        self.optimizer.step()
        
        self.losses.append(loss.item())

    def update_target_network(self) -> None:
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(state_dict=target_net_state_dict)

    def plot_metrics(self, show_result: bool = False) -> None:
        # Set the font to Times New Roman
        plt.rcParams["font.family"] = "Times New Roman"

        epochs = list(range(1, len(self.epoch_durations) + 1))

        # Plot epoch Durations
        plt.figure()
        plt.title('Epoch Duration')
        durations_t = torch.tensor(data=self.epoch_durations, dtype=torch.float)
        plt.xlabel('Epoch')
        plt.ylabel('Duration')
        plt.plot(epochs, durations_t.numpy(), label='Duration')
        if len(durations_t) >= 50:
            means = durations_t.unfold(dimension=0, size=50, step=1).mean(dim=1).view(-1)
            means = torch.cat((torch.zeros(49), means))
            plt.plot(epochs, means.numpy(), label='Moving Average')
        plt.legend()
        plt.savefig(f'{self.path}/epoch_duration.svg')
        plt.show()

        # Plot epoch Rewards
        plt.figure()
        plt.title('Epoch Reward')
        rewards_t = torch.tensor(data=self.epoch_rewards, dtype=torch.float)
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.plot(epochs, rewards_t.numpy(), label='Reward')
        if len(rewards_t) >= 50:
            means = rewards_t.unfold(dimension=0, size=50, step=1).mean(dim=1).view(-1)
            means = torch.cat((torch.zeros(49), means))
            plt.plot(epochs, means.numpy(), label='Moving Average')
        plt.legend()
        plt.savefig(f'{self.path}/epoch_reward.svg')
        plt.show()

        # Plot Losses
        plt.figure()
        plt.title('Loss')
        plt.xlabel('Optimization Step')
        plt.ylabel('Loss')
        plt.plot(self.losses, label='Loss')
        plt.legend()
        plt.savefig(f'{self.path}/loss.svg')
        plt.show()

        # Plot Epsilon Decay
        plt.figure()
        plt.title('Epsilon Decay')
        plt.xlabel('Step')
        plt.ylabel('Epsilon')
        plt.plot(self.epsilons, label='Epsilon')
        plt.legend()
        plt.savefig(f'{self.path}/epsilon_decay.svg')
        plt.show()

    def run_epoch(self, training: bool = True) -> float:
        state, _ = self.env.reset()
        state = self.preprocess_frame(frame=state)
        state = np.stack([state] * 4, axis=0)  # Stack 4 frames for the initial state
        state = torch.from_numpy(state).unsqueeze(dim=0).to(device=device, dtype=torch.float32)

        total_reward = 0.0
        for t in count():
            action = self.select_action(state=state)
            observation, reward, terminated, truncated, _ = self.env.step(action=action.item())
            reward = torch.tensor(data=[reward], device=device)
            total_reward += reward.item()
            done = terminated or truncated

            if not done:
                next_frame = self.preprocess_frame(frame=observation)
                next_state = np.append(state.cpu().numpy()[0, 1:], np.expand_dims(next_frame, axis=0), axis=0)
                next_state = torch.from_numpy(next_state).unsqueeze(dim=0).to(device=device, dtype=torch.float32)
            else:
                next_state = None

            if training:
                self.memory.push(state, action, next_state, reward)
                self.optimize_model()
                self.update_target_network()

            state = next_state

            if done:
                if training:
                    self.epoch_durations.append(t + 1)
                    self.epoch_rewards.append(total_reward)
                    self.save_metrics()
                break

        return total_reward

    def train(self, num_epochs: int) -> None:
        for i_epoch in range(1, num_epochs + 1):
            print(f"Start Training epoch {i_epoch}/{num_epochs}")
            self.run_epoch(training=True)
            print(f"End Training epoch {i_epoch}/{num_epochs}")
        print('Complete')
        self.plot_metrics(show_result=True)
        average_loss = np.mean(self.losses)
        print(f"Average loss over all {num_epochs} training epochs: {average_loss}")
        plt.ioff()