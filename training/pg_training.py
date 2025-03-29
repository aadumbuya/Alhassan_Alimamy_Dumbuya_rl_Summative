import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv


def training_ppo(env=None,method="MlpPolicy",n_steps=2048,ent_coef=0.05,gamma=0.99,learning_rate=1e-4):
    # Initialize environment
    assert env is not None,"Environment is not defined"
    env = Monitor(env)  # Wrap for logging
    vec_env = DummyVecEnv([lambda: env])  # Vectorized environment

    # Hyperparameter Optimization
    model = PPO(method,
    vec_env,
    n_steps=n_steps,  # Larger batch for better updates
    ent_coef=ent_coef,  # Less entropy for more exploitation
    gamma=gamma,  # Discount factor
    learning_rate=learning_rate,  # Lower LR for stable training
    verbose=1,
    tensorboard_log="./ppo_auv_log/")

    # Callbacks: Model Evaluation & Checkpoints
    eval_callback = EvalCallback(vec_env,best_model_save_path="./ppo_auv_best_model/",log_path="./ppo_auv_logs/",eval_freq=5000,deterministic=True,render=False)

    checkpoint_callback = CheckpointCallback(save_freq=25000,  # Save model every 25k steps
    save_path="./ppo_auv_checkpoints/",
    name_prefix="rl_model")

    # Train Model
    model.learn(total_timesteps=100000, callback=[eval_callback, checkpoint_callback])

    # Save Final Model
    model.save("ppo_auv_navigation")
    return model


