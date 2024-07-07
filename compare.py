num_episodes = 1

from baseline import run_baseline
from main import finalModels

def compare() -> None:
    print("Running the baseline model...")
    # multiply by 3 to get the same number of episodes as the final model due to the
    # fact that the baseline model executes only 1 episode per run and the final model
    # executes 3 episodes per run
    baseline_episodes, baseline_score = run_baseline(num_episodes=num_episodes*3)
    print("Running the final model...")
    final_episodes, final_score = finalModels(num_episodes=num_episodes)
    
    print(f"Baseline model: Average score over {baseline_episodes} episodes = {baseline_score}")
    print(f"Final model: Average score over {final_episodes} episodes = {final_score}")
    
    if final_score > baseline_score:
        print("Final model outperformed the baseline model")
    elif final_score < baseline_score:
        print("Final model underperformed the baseline model")
    else:
        print("Final model performed equally to the baseline model")

if __name__ == "__main__":
    compare()