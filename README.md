# SnakeAI: Deep Q-Learning with PyTorch

A reinforcement learning agent that teaches itself to play Snake using a Linear Q-Network (Deep Q-Learning).

## 🧠 The Architecture (Neural Network)
The AI does not know the rules of Snake. It learns by "looking" at the state of the game and adjusting its neural weights to maximize a reward function.

* **Input Layer (11 Neurons):** Perception (Danger Straight/Right/Left, Current Direction, Food Location).
* **Hidden Layer (256 Neurons):** ReLU activation for non-linear decision making.
* **Output Layer (3 Neurons):** Action probabilities (Straight, Right, Left).

## 📐 The Math (Bellman Equation)
The model is trained using the **Bellman Equation** to minimize the loss between the predicted Q-value and the target Q-value:

$$Q_{new}(s,a) = R + \gamma \max(Q(s', a'))$$

* **$R$**: Reward (Food: +10, Death: -10, Else: 0)
* **$\gamma$**: Discount factor (0.9), prioritizing future rewards.
* **Optimization**: Adam Optimizer using Mean Squared Error (MSE) Loss.

## 🚀 How to Run
1.  Install dependencies:
    ```bash
    pip install torch pygame matplotlib numpy
    ```
2.  Run the training agent:
    ```bash
    python agent.py
    ```
3.  Watch the graph! The agent will start random (epsilon-greedy) and gradually improve as `n_games` increases.