import wandb
import pandas as pd
from pathlib import Path
import time
import os
import sys
import tensorflow as tf
import argparse

from utils.gpu_utils import configure_gpu
from models.environment import EnergyEnv
from models.dqn_agent import DQNAgent
from models.ppo_agent import PPOAgent
from train.trainer import train_dqn, train_ppo  # Assuming you'll create train_ppo

def preprocess_data(data):
    """Preprocess the data with appropriate column names"""
    # Rename columns
    column_mapping = {
        'Tou Tariff': 'Tou_Tariff',
        'H2 Tariff': 'H2_Tariff'
    }
    
    # Rename only if the columns exist
    for old_name, new_name in column_mapping.items():
        if old_name in data.columns:
            data = data.rename(columns={old_name: new_name})
    
    return data

def setup_gpu():
    """Setup GPU environment with error handling"""
    try:
        # Configure GPU
        if not configure_gpu():
            print("Warning: GPU configuration failed, falling back to CPU")
        
        # Print GPU information
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            print(f"GPU Device: {gpu.device_type} - {gpu.name}")
            
        # Check CUDA availability
        if not tf.test.is_built_with_cuda():
            print("Warning: TensorFlow was not built with CUDA support")
            
    except Exception as e:
        print(f"Error setting up GPU: {str(e)}")
        print("Falling back to CPU")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train EMS agent with selected algorithm')
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'ppo'], default='dqn',
                      help='Algorithm to use for training (dqn or ppo)')
    parser.add_argument('--episodes', type=int, default=200,
                      help='Number of episodes for training')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Setup GPU environment
    setup_gpu()

    # Create checkpoints directory
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    try:
        # Load training data
        data_path = 'training_data.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found at {data_path}")
            
        data = pd.read_csv(data_path)
        data = preprocess_data(data)
        
        # Initialize environment
        env = EnergyEnv(data)
        
        # Get state and action sizes from environment
        state_size = env.state_size()
        action_size = 3  # 0: Do nothing, 1: Battery operations, 2: H2 operations
        
        # Initialize agent based on selected algorithm
        if args.algorithm.lower() == 'dqn':
            agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                initial_epsilon=1.0,
                epsilon_min=0.01,
                episodes=args.episodes,
                case_name="DQN"
            )
            print("\nStarting training for EMS DQN Agent")
            training_function = train_dqn
        else:  # PPO
            agent = PPOAgent(
                state_size=state_size,
                action_size=action_size,
                case_name="PPO"
            )
            print("\nStarting training for EMS PPO Agent")
            training_function = train_ppo

        print("=" * 80)
        
        # Record start time
        start_time = time.time()

        try:
            # Train the agent with selected algorithm
            training_function(env, agent, args.episodes)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            wandb.finish()
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            wandb.finish()
            raise

        finally:
            # Calculate and display total training time
            end_time = time.time()
            total_time = end_time - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print("\nTraining completed")
            print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            print("=" * 80)
            
            wandb.finish()

    except Exception as e:
        print(f"Critical error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()