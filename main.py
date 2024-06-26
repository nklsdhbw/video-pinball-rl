import gymnasium as gym
import torch
from agent import DQNAgent
from dqn import DQN
import numpy as np
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env = gym.make("ALE/VideoPinball-v5", render_mode="human")
    agent = DQNAgent(env=env)
    
    # Load the trained model
    agent.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=torch.device('cpu')))
    agent.policy_net.eval()
    
    # Interact with the environment using the loaded model
    state, _ = env.reset()
    state = agent.preprocess_frame(frame=state)
    state = np.stack([state] * 4, axis=0)  # Stack 4 frames for the initial state
    state = torch.from_numpy(state).unsqueeze(dim=0).to(device=device, dtype=torch.float32)

    for t in count():
        action = agent.select_action(state=state)
        observation, reward, terminated, truncated, _ = env.step(action=action.item())
        reward = torch.tensor(data=[reward], device=device)
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

if __name__ == "__main__":
    main()