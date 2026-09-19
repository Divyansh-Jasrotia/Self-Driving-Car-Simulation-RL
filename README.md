# 🚗 Autonomous Navigation Simulation — Deep Q-Learning (RL)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Kivy](https://img.shields.io/badge/GUI-Kivy-3399db.svg)](https://kivy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An autonomous vehicle navigation simulator built in **Python** using **PyTorch** and **Kivy**. The agent learns continuous optimal driving, obstacle avoidance, and goal-directed navigation in real time using **Deep Q-Networks (DQN)** coupled with **Experience Replay**.

Users can draw obstacles and roads interactively on the canvas, observing how the neural network adapts its steering decisions through reinforcement learning.

---

## 📌 Architecture & RL Formulation

The problem is modeled as a continuous Markov Decision Process (MDP) where the agent interacts with a 2D physics sandbox:

```
           +---------------------------------------------+
           |                Kivy Canvas                  |
           |   (Obstacles / Sand Map + Target Goal)      |
           +---------------------------------------------+
                                  |
                   State Vector S_t (5 dimensions)
                   [sensor1, sensor2, sensor3, theta, -theta]
                                  v
+-----------------------------------------------------------------+
|                    Deep Q-Network (PyTorch)                     |
|                                                                 |
|   Input (5) ---> Linear(5, 30) ---> ReLU ---> Linear(30, 3)     |
|                                                                 |
|                  Output: Q(s, a_0), Q(s, a_1), Q(s, a_2)        |
+-----------------------------------------------------------------+
                                  |
                  Action Selection (Softmax Policy)
                       a_t in {0°, +20°, -20°}
                                  v
+-----------------------------------------------------------------+
|                    Experience Replay Buffer                     |
|                                                                 |
|   Stores: (s_t, s_{t+1}, a_t, r_t)  [Capacity: 10,000]          |
|   Loss: Smooth L1 (Huber Loss) | Optimizer: Adam (lr=0.001)     |
+-----------------------------------------------------------------+
```

### 1. State Space ($S \in \mathbb{R}^5$)
At every frame, the car samples 5 normalized features:
- **`sensor1`**: Proximity sensor ray detecting sand/obstacle density in the front-left.
- **`sensor2`**: Proximity sensor ray detecting sand/obstacle density directly ahead.
- **`sensor3`**: Proximity sensor ray detecting sand/obstacle density in the front-right.
- **`orientation`**: Angular discrepancy between the car's current heading vector and the destination vector.
- **`-orientation`**: Inverted heading angle providing symmetric spatial context to the neural network.

### 2. Action Space ($A$)
The steering policy chooses among 3 discrete rotational actions:
| Action Index | Rotation Angle | Trajectory Effect |
|:---:|:---:|:---|
| `0` | $0^\circ$ | Continue straight ahead |
| `1` | $+20^\circ$ | Steer clockwise (turn right) |
| `2` | $-20^\circ$ | Steer counter-clockwise (turn left) |

### 3. Reward Function ($R$)
The reward function is engineered to balance goal convergence and hazard avoidance:
- **Obstacle / Sand Penalty**: Entering user-drawn sand or colliding with obstacles delivers a negative reward ($-1.0$), forcing the car to seek clear roads.
- **Goal Progress Reward**: Moving closer to the current target awards positive incremental feedback ($+0.1$), while deviating away applies a mild penalty ($-0.2$).
- **Perimeter Bounds**: Approaching canvas boundaries applies proximity penalties to prevent boundary trapping.

### 4. Experience Replay & Policy Optimization
- **Replay Memory**: Experiences $(s_t, a_t, r_t, s_{t+1})$ are pushed to a memory buffer with a capacity of $10,000$ transitions. Random batch sampling ($N=32$) breaks temporal correlation between consecutive states.
- **Temporal Difference Learning**:
  $$\mathcal{L} = \text{Smooth}_{L1}\left(Q(s_t, a_t) - \left[r_t + \gamma \max_{a'} Q(s_{t+1}, a')\right]\right)$$
  with discount factor $\gamma = 0.9$.
- **Action Selection**: Employs a Softmax (Boltzmann) policy over output Q-values with a temperature factor to smoothly manage exploration vs. exploitation.

---

## 🎮 Interactive Features

- **Dynamic Sand Painting**: Click and drag your mouse across the canvas to draw obstacles or terrain boundaries in real time.
- **Dynamic Goal Switching**: The vehicle navigates toward Goal A (top-left). Upon reaching it, the destination automatically toggles to Goal B (bottom-right) to enforce two-way route planning.
- **Live Memory Management**:
  - **Clear**: Wipes all drawn obstacles from the canvas.
  - **Save**: Serializes the trained PyTorch network weights into `last_brain.pth`.
  - **Load**: Restores previously trained model weights for immediate inference.

---

## 📂 Repository Structure

```
.
├── agent.py          # PyTorch Deep Q-Network, ReplayMemory, and DQN agent
├── car.kv            # Kivy UI layout defining canvas widgets, car body, and buttons
├── main.py           # Simulation engine, vehicle kinematics, sensor updates & game loop
├── requirements.txt  # Python package dependencies
└── README.md         # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9, 3.10, or 3.11
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Divyansh-Jasrotia/Self-Driving-Car-Simulation-RL.git
   cd Self-Driving-Car-Simulation-RL
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the simulation:**
   ```bash
   python main.py
   ```

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch (`torch`, `torch.nn`, `torch.optim`)
- **Simulation GUI**: Kivy (`App`, `Widget`, `Graphics`, `Clock`)
- **Scientific Computing**: NumPy, Matplotlib

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.