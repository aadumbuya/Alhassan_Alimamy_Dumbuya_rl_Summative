# AUV Navigation with Reinforcement Learning

## Project Overview
This project trains an Autonomous Underwater Vehicle (AUV) to navigate through an environment while avoiding obstacles and optimizing energy consumption. Using reinforcement learning, the agent learns efficient movement strategies to reach a goal while managing limited battery life.

## Environment
- **State Representation:** (x, y, z) position, battery level, and distance to goal.
- **Actions:** Forward, Backward, Left, Right, Ascend, Descend (Discrete action space).
- **Reward Function:** Reward for moving towards the goal, penalty for hitting obstacles, and gradual battery depletion.

## Implemented Methods
1. **DQN (Deep Q-Network)**
   - Experience replay and target networks for stability.
   - Optimized hyperparameters: gamma=0.99, learning rate=1e-4, replay buffer size=50,000.
   - Outperformed PPO in generalization and stability.

2. **PPO (Proximal Policy Optimization)**
   - Policy gradient method with entropy regularization.
   - Stable training with gamma=0.99, learning rate=1e-4.
   - Struggled with unseen environments and had higher variance.

## Results
- **DQN performed better**, achieving faster convergence and stable performance across episodes.
- **PPO had higher variance**, making it less reliable in unseen states.
- **Generalization Tests:** DQN handled unseen conditions better due to experience replay.

## Visualization
- Included plots for cumulative rewards and loss curves.
- Rendered OpenCV-based visual representation of the AUV environment.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python main.py`

## Future Improvements
- Experimenting with hybrid RL approaches.
- Enhancing state representation with sensor fusion.
- Fine-tuning PPO for better generalization.
