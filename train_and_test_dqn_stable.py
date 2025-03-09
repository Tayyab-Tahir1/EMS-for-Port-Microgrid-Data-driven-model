import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import argparse
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

# Import your custom EnergyEnv from models/environment.py
from models.environment import EnergyEnv

# -------------------------------
# Gymnasium Wrapper for EnergyEnv
# -------------------------------
class EnergyEnvGym(gym.Env):
    """
    Wraps your custom EnergyEnv (which is not Gymnasium-compatible) into a Gymnasium environment.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, data):
        super(EnergyEnvGym, self).__init__()
        # Instantiate your custom environment
        self.env = EnergyEnv(data)
        # Define observation space (state is 8-dimensional and normalized to [0,1])
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        # Define action space: 3 discrete actions
        self.action_space = spaces.Discrete(3)
        
    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return np.array(obs, dtype=np.float32), {}
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return np.array(state, dtype=np.float32), reward, done, False, info
    
    def render(self, mode="human"):
        soc = self.env.soc / self.env.battery_capacity * 100
        print(f"Step: {self.env.current_step}, Battery SoC: {soc:.2f}%")

# -------------------------------
# Wandb Callback for Stable-Baselines3
# -------------------------------
class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        wandb.log({"timesteps": self.num_timesteps, "reward": reward})
        return True

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_data(data):
    """
    Rename dataset columns so they match what your EnergyEnv expects.
    For example, if your dataset uses 'Tou Tariff' and 'H2 Tariff', they are renamed.
    """
    rename_mapping = {
        'Tou Tariff': 'Tou_Tariff',
        'H2 Tariff': 'H2_Tariff'
    }
    data.rename(columns=rename_mapping, inplace=True)
    return data

# -------------------------------
# Main Function (Fine-Tuned Version)
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train and Test Fine-Tuned DQN on Energy Management Environment")
    parser.add_argument("--mode", type=str, choices=["train", "test", "both"], default="both",
                        help="Mode to run: train, test, or both")
    # Increase the number of training timesteps for better convergence
    parser.add_argument("--timesteps", type=int, default=500000,
                        help="Total timesteps for training")
    parser.add_argument("--test_episodes", type=int, default=10,
                        help="Number of episodes for testing")
    args = parser.parse_args()

    # Initialize wandb (ensure you've run `wandb login`)
    wandb.init(project="EMS_DQN_Finetuned", name="DQN_FineTune_Run")
    
    # Load dataset from training_data.csv
    dataset_path = 'training_data.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    data = pd.read_csv(dataset_path)
    data = preprocess_data(data)
    
    # Create Gymnasium environment wrapper around your EnergyEnv
    env = EnergyEnvGym(data)
    check_env(env, warn=True)
    
    # Define model save path
    model_path = "dqn_energy_model_finetuned"
    
    # Define custom policy parameters: you can experiment with the network architecture here
    policy_kwargs = dict(net_arch=[128, 128])  # Two hidden layers with 256 neurons each
    
    if args.mode in ["train", "both"]:
        print("Training mode selected.")
        # Fine-tuning hyperparameters
        model = DQN(
            "MlpPolicy", 
            env, 
            learning_rate=1e-4,            # Lower learning rate for stability
            buffer_size=50000,              #100000,            # Increase replay buffer size
            exploration_fraction=0.1,       #0.2,      # Longer exploration phase
            exploration_final_eps=0.05,      # Final epsilon value for exploitation
            gamma=0.99,                    # Discount factor
            batch_size=64,                 # Batch size
            policy_kwargs=policy_kwargs,   # Custom network architecture
            verbose=1, 
            tensorboard_log="./tb_logs_finetuned/"
        )
        # Train the model with the wandb callback for logging
        model.learn(total_timesteps=args.timesteps, callback=WandbCallback())
        model.save(model_path)
        print(f"Model saved as {model_path}.zip")
    else:
        if os.path.exists(model_path + ".zip"):
            model = DQN.load(model_path, env=env)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}.zip")
    
    if args.mode in ["test", "both"]:
        print("Testing mode selected.")
        for ep in range(args.test_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            print(f"\nStarting test episode {ep+1}:")
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, trunc, info = env.step(action)
                total_reward += reward
                env.render()  # Render current step info
                if done or trunc:
                    break
            print(f"Episode {ep+1} finished with total reward: {total_reward:.2f}")
    
    wandb.finish()

if __name__ == "__main__":
    main()
