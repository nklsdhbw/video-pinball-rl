import gymnasium as gym
import torch
import os

if os.path.basename(os.getcwd()) != 'video-pinball-rl':
    os.chdir("./video-pinball-rl")
from agent import DQNAgent

def train_model() -> None:
    env = gym.make("ALE/VideoPinball-v5", render_mode="rgb_array")
    agent = DQNAgent(env=env)
    
    if torch.cuda.is_available():
        num_epochs = 100
    else:
        num_epochs = 1
    
    agent.train(num_epochs=num_epochs)
    
    # Save the trained model
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    print("Model saved")

if __name__ == "__main__":
    train_model()
