import gymnasium as gym
import torch
import os

if os.path.basename(os.getcwd()) != 'video-pinball-rl':
    os.chdir("./video-pinball-rl")
from agent import DQNAgent

def train_model(num_epochs=10, batch_size=128) -> None:
    env = gym.make("ALE/VideoPinball-v5", render_mode="rgb_array")
    agent = DQNAgent(env=env, batch_size=batch_size, gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000, tau=0.0005, lr=1e-4, memory_size=50000, path=".")
    
    agent.train(num_epochs=num_epochs)
    
    # Save the trained model
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    print("Model saved")

if __name__ == "__main__":
    train_model()