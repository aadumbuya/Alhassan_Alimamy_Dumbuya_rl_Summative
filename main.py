import gym
from environment.custom_env import AUVNavigationEnv  # Ensure your custom environment is in place
from training.dqn_training import training_dqn  # Import DQN training method
from training.pg_training import training_ppo  # Import PPO training method
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

def main():
    # Create the environment
    env = AUVNavigationEnv()

    # Train DQN Model
    print("Training DQN Model...")
    dqn_model = training_dqn(env=env,method="MlpPolicy",buffer_size=50000,learning_starts=10000,batch_size=64,tau=1.0,gamma=0.99,learning_rate=1e-4,exploration_fraction=0.1,exploration_final_eps=0.05,train_freq=4,target_update_interval=1000)

    # Evaluate DQN Model
    print("Evaluating DQN Model...")
    dqn_mean_reward, dqn_std_reward = evaluate_policy(dqn_model, env, n_eval_episodes=10)
    print(f"DQN Model Evaluation - Mean Reward: {dqn_mean_reward}, Standard Deviation of Reward: {dqn_std_reward}")

    # Train PPO Model
    print("Training PPO Model...")
    ppo_model = training_ppo(env=env,method="MlpPolicy",n_steps=2048,ent_coef=0.05,gamma=0.99,learning_rate=1e-4)

    # Evaluate PPO Model
    print("Evaluating PPO Model...")
    ppo_mean_reward, ppo_std_reward = evaluate_policy(ppo_model, env, n_eval_episodes=10)
    print(f"PPO Model Evaluation - Mean Reward: {ppo_mean_reward}, Standard Deviation of Reward: {ppo_std_reward}")

    # Compare the performance of DQN and PPO
    print("Comparison of DQN and PPO Performance:")
    print(f"DQN - Mean Reward: {dqn_mean_reward}, Standard Deviation: {dqn_std_reward}")
    print(f"PPO - Mean Reward: {ppo_mean_reward}, Standard Deviation: {ppo_std_reward}")

if __name__ == "__main__":
    main()



