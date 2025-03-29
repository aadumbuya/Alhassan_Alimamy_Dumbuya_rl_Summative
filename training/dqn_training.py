import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

def training_dqn(env=None,method="MlpPolicy",buffer_size=50000,learning_starts=10000,batch_size=64,tau=1.0,gamma=0.99,learning_rate=1e-4,exploration_fraction=0.1,exploration_final_eps=0.05,train_freq=4,target_update_interval=1000):
    # Initialize environment
    assert env is not None,"Environment is not defined"
    env = Monitor(env)  # Wrap for logging
    vec_env = DummyVecEnv([lambda: env])  # Vectorized environment

    # DQN Model with Hyperparameters
    model = DQN(method,vec_env,
        buffer_size=buffer_size,  # Larger replay buffer
        learning_starts=learning_starts,  # Start learning after 10k steps
        batch_size=batch_size,  # Mini-batch size
        tau=tau,  # Soft update factor for target network
        gamma=gamma,  # Discount factor
        learning_rate=learning_rate,  # Lower learning rate for stability
        exploration_fraction=exploration_fraction,  # Fraction of exploration phase
        exploration_final_eps=exploration_final_eps,  # Minimum exploration rate
        train_freq=train_freq,  # Train every 4 steps
        target_update_interval=target_update_interval,  # Update target network every 1k steps
        verbose=1,
        tensorboard_log="./dqn_auv_log/"
    )

    # Callbacks: Model Evaluation & Checkpoints
    eval_callback = EvalCallback(vec_env,best_model_save_path="./dqn_auv_best_model/",log_path="./dqn_auv_logs/",eval_freq=5000,deterministic=True,render=False)

    checkpoint_callback = CheckpointCallback(save_freq=25000,  # Save model every 25k steps
    save_path="./dqn_auv_checkpoints/",
    name_prefix="rl_model"
    )

    # Train Model
    model.learn(total_timesteps=100000, callback=[eval_callback, checkpoint_callback])

    # Save Final Model
    model.save("dqn_auv_navigation")
    return model

    