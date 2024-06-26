import gymnasium as gym
import torch
from agent import DQNAgent

def main():
    env = gym.make("ALE/VideoPinball-v5", render_mode="human")
    agent = DQNAgent(env=env)
    
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 50
    
    agent.train(num_episodes=num_episodes)

if __name__ == "__main__":
    main()
