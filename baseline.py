import gymnasium as gym
import numpy as np
from typing import Tuple

def run_baseline(num_episodes=3) -> Tuple[int, float]:
    env = gym.make("ALE/VideoPinball-v5", render_mode="rgb_array")
    num_episodes = 3
    scores = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        counter = 0
        score = 0
        last_ball_position = None
        ball_stuck_counter = 0

        actions = [5, 1, 7, 6, 3, 4, 7, 8, 5, 1]

        while not done:
            action = 0
            if counter in range(0, 10):
                action = actions[counter]
            else:
                action = 2 if counter % 2 == 0 else 6

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            score += reward
            counter += 1

            # Check if the ball is stuck
            current_ball_position = info.get('ball_x', None), info.get('ball_y', None)
            if current_ball_position == last_ball_position:
                ball_stuck_counter += 1
            else:
                ball_stuck_counter = 0

            last_ball_position = current_ball_position

            if ball_stuck_counter > 30000:  # 30000 frames is roughly 5 seconds as rgb_array mode returns 5000 frames per second
                done = True

            if terminated or truncated:
                done = True

        scores.append(score)
        env.reset()

    env.close()
    average_score = np.mean(scores)
    print(scores)
    print(f"Average score over {num_episodes} episodes: {average_score}")
    return num_episodes, average_score
if __name__ == "__main__":
    run_baseline()
