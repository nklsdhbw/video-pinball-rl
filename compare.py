
import os
if os.path.basename(os.getcwd()) != 'video-pinball-rl':
    os.chdir("./video-pinball-rl")
num_epochs = 1
from baseline import run_baseline
from models import models

def compare() -> None:
    print("Running the baseline model...")
    # multiply by 3 to get the same number of episodes as the final model due to the
    # fact that the baseline model executes only 1 episode per epoch and the final model
    # executes 3 episodes per run
    baseline_epochs, baseline_score = run_baseline(num_epochs=num_epochs*3)
    print("Running the final model...")
    final_epochs, final_score = models(num_epochs=num_epochs)
    
    print(f"Baseline model: Average score over {baseline_epochs} episodes = {baseline_score}")
    print(f"Final model: Average score over {final_epochs*3} episodes = {final_score}")
    
    if final_score > baseline_score:
        print("Final model outperformed the baseline model")
    elif final_score < baseline_score:
        print("Final model underperformed the baseline model")
    else:
        print("Final model performed equally to the baseline model")

if __name__ == "__main__":
    compare()