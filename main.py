import gymnasium as gym
import torch
from agent import DQNAgent
from dqn import DQN
import numpy as np
from itertools import count
import os
from typing import List, Optional, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def finalModels(num_episodes) -> None:
    files = os.listdir()
    if "dqn_model.pth" not in files:
        print("Model not found. Please train the model first via 'python train.py'")
        print("Exiting...")
        return
    else:
        print("Create an environment and agent...")
        env = gym.make("ALE/VideoPinball-v5", render_mode="rgb_array")
        agent = DQNAgent(env=env)
        print("Environment and agent successfully created")
        
        # Load the trained model
        print("Load the trained model...")
        agent.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=torch.device('cpu')))
        agent.policy_net.eval()
        
        scores: List[float] = []
        print(f"Start testing the model over {num_episodes} episodes...")
        for episode in range(num_episodes):
            state, _ = env.reset()
            state = agent.preprocess_frame(frame=state)
            state = np.stack([state] * 4, axis=0)  # Stack 4 frames for the initial state
            state = torch.from_numpy(state).unsqueeze(dim=0).to(device=device, dtype=torch.float32)

            total_reward = 0.0

            for t in count():
                action = agent.select_action(state=state)
                observation, reward, terminated, truncated, _ = env.step(action=action.item())
                reward = torch.tensor(data=[reward], device=device)
                total_reward += reward.item()
                done = terminated or truncated

                if not done:
                    next_frame = agent.preprocess_frame(frame=observation)
                    next_state = np.append(state.cpu().numpy()[0, 1:], np.expand_dims(next_frame, axis=0), axis=0)
                    next_state = torch.from_numpy(next_state).unsqueeze(dim=0).to(device=device, dtype=torch.float32)
                else:
                    next_state = None

                state = next_state

                if done:
                    break

            scores.append(total_reward)
            print(f"Episode {episode + 1}: Score = {total_reward}")

        print("Testing complete")
        average_score = np.mean(scores)
        print(f"Average score over {num_episodes} episodes: {average_score}")
        print("Exiting...")
        return num_episodes, average_score
if __name__ == "__main__":
    finalModels()
