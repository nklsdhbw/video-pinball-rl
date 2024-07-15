# 🕹️ Atari Video Pinball Reinforcement Learning Agent

  

This project implements a Deep Q-Network (DQN) agent to play Atari Video Pinball using PyTorch.

  

## 📖 Game Description

  

Atari Video Pinball is a digital pinball game where the objective is to score as many points as possible by manipulating flippers to hit a ball into various targets and bumpers. The game is a part of the Atari 2600 suite of games and is commonly used in reinforcement learning research.

  

## ⚙️ Actions

  

The action space for Atari Video Pinball consists of 9 discrete actions:

  

	1. NOOP: No operation

	2. FIRE: Fire the ball

	3. UP: Move up

	4. RIGHT: Move right

	5. LEFT: Move left

	6. DOWN: Move down

	7. UPRIGHT: Move up and right

	8. UPLEFT: Move up and left

	9. DOWNRIGHT: Move down and right

	10. DOWNLEFT: Move down and left

  

These actions allow the agent to control the flippers and manipulate the ball's trajectory to maximize the score.

  

## 🧮 States

  

The state representation in this implementation is based on the game screen frames. Each state is a stack of 4 consecutive frames, each resized to 84x84 pixels and converted to grayscale. This helps the agent to capture the motion dynamics of the game.

  

### 🛠️ Preprocessing

  

Each frame undergoes the following preprocessing steps:

  

	1. Convert the frame to grayscale.

	2. Resize the frame to 84x84 pixels.

	3. Normalize pixel values to the range [0, 1].

  

The final state is a 4x84x84 tensor, representing a stack of 4 processed frames.

  

## 📈 Q-Function

  

The Q-function is a neural network that estimates the expected future rewards for each action given a state. The Q-values are updated using the Bellman equation:

  

$Q(s, a) = r + \gamma \max_{a'} Q(s', a')$

  

Where:

- $Q(s, a)$ is the Q-value for state \( s \) and action \( a \).

- $r$ is the reward received after taking action \( a \) in state \( s \).

- $\gamma$ is the discount factor.

- $s'$ is the next state.

- $a'$ is the next action.

  

The loss function used to train the Q-network is the Huber loss:

  

$L(\theta) = \mathbb{E}_{s, a, r, s'} \left[ \left( y - Q(s, a; \theta) \right)^2 \right]$

  

Where:

- $y = r + \gamma \max_{a'} Q(s', a'; \theta^{-})$ is the target Q-value.

- $\theta$ are the parameters of the policy network.

- $\theta^{-}$ are the parameters of the target network.

  

## 🚀 Getting started

### 🐍 Python

[](https://github.com/nklsdhbw/election-manifestos-analysis#-python)

Download and install  [Python 3.9](https://www.python.org/downloads/)  

### ⬇️ Download repository & run the application
Run the following command in your terminal to clone the repo

	git clone https://github.com/nklsdhbw/video-pinball-rl.git

To run this project, you'll need Python and the required dependencies. You can install the dependencies using:

  



	python3.9 -m venv atari

If you're using Windows then run

	source atari\Scripts\activate

If you're using MacOS or Linux then run

	source atari/bin/activate
Now install all the required packages from `requirements.txt` by running
	
	pip install -r requirements.txt

#### Training
<b> Warning!</b> Running this script will override the current model and their corresponding files and plots. 

First set the hyperparamters in `train.py`
Then execute the script via the following command

	python train.py
The plots, the underlying csv-files for the epsilon decay, the episode rewards & durations and the losses as well as the final DQN model are automatically generated and saved.
#### Comparison
To compare the results of the Baseline method with the Double DQN, you'll need to train it first. When this step is done, run the following command

	python comparison.py