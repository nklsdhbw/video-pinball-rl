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
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.render_mode = env.render_mode
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
        self.episode_rewards = []
        self.losses = []
        self.epsilons = []

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        # Convert to grayscale and resize
        frame = cv2.cvtColor(src=frame, code=cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(src=frame, dsize=(84, 84))
        frame = frame / 255.0
        return frame

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        self.epsilons.append(eps_threshold)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(dim=1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def optimize_model(self) -> None:
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
        
        self.losses.append(loss.item())

    def update_target_network(self) -> None:
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(state_dict=target_net_state_dict)

    def plot_metrics(self, show_result: bool = False) -> None:
        episodes = list(range(1, len(self.episode_durations) + 1))

        # Plot Episode Durations
        plt.figure()
        plt.title('Episode Duration')
        durations_t = torch.tensor(data=self.episode_durations, dtype=torch.float)
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(episodes, durations_t.numpy())
        plt.xticks(episodes)  # Set x-ticks to episode numbers
        if len(durations_t) >= 100:
            means = durations_t.unfold(dimension=0, size=100, step=1).mean(dim=1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(episodes, means.numpy())
        plt.savefig('episode_duration.png')
        plt.show()

        # Plot Episode Rewards
        plt.figure()
        plt.title('Episode Reward')
        rewards_t = torch.tensor(data=self.episode_rewards, dtype=torch.float)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(episodes, rewards_t.numpy())
        plt.xticks(episodes)  # Set x-ticks to episode numbers
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(dimension=0, size=100, step=1).mean(dim=1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(episodes, means.numpy())
        plt.savefig('episode_reward.png')
        plt.show()

        # Plot Losses
        plt.figure()
        plt.title('Loss')
        plt.xlabel('Optimization Step')
        plt.ylabel('Loss')
        plt.plot(self.losses)
        plt.savefig('loss.png')
        plt.show()

        # Plot Epsilon Decay
        plt.figure()
        plt.title('Epsilon Decay')
        plt.xlabel('Step')
        plt.ylabel('Epsilon')
        plt.plot(self.epsilons)
        plt.savefig('epsilon_decay.png')
        plt.show()

    def run_episode(self, training: bool = True) -> float:
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
                    self.episode_durations.append(t + 1)
                    self.episode_rewards.append(total_reward)
                break

        return total_reward

    def train(self, num_episodes: int) -> None:
        for i_episode in range(1, num_episodes + 1):
            print(f"Start Training episode {i_episode}/{num_episodes}")
            self.run_episode(training=True)
            print(f"End Training episode {i_episode}/{num_episodes}")
        print('Complete')
        self.plot_metrics(show_result=True)
        plt.ioff()