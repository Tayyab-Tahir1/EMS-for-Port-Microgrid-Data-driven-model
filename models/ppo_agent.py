import os
import wandb
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.losses import Huber

class PPOAgent:
    def __init__(
        self,
        state_size,
        action_size,
        case_name="default",
        clip_ratio=0.1,
        policy_learning_rate=1e-4,
        value_learning_rate=5e-5,
        gamma=0.99,
        lambda_gae=0.98,
        entropy_beta=0.005
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.entropy_beta = entropy_beta
        self.case_name = case_name
        self.best_reward = float('-inf')
        
        self.checkpoint_dir = Path(f'checkpoints/{case_name}')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Set XLA GPU flags
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/apps/cuda/11.2.2'
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
        
        with tf.device('/GPU:0'):
            self.policy_network = self._build_policy_network()
            self.value_network = self._build_value_network()
            
            self.policy_optimizer = optimizers.Adam(learning_rate=policy_learning_rate, clipnorm=1.0)
            self.value_optimizer = optimizers.Adam(learning_rate=value_learning_rate, clipnorm=1.0)
        
        self.init_wandb()
    
    def init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project="Energy Management System",
            name=f"EMS-Port-{self.case_name}",
            config={
                "policy_learning_rate": self.policy_optimizer.learning_rate.numpy(),
                "value_learning_rate": self.value_optimizer.learning_rate.numpy(),
                "clip_ratio": self.clip_ratio,
                "gamma": self.gamma,
                "lambda_gae": self.lambda_gae,
                "entropy_beta": self.entropy_beta,
                "case_name": self.case_name
            }
        )
        
        # Configure metrics
        wandb.define_metric("Episode")
        wandb.define_metric("Policy Loss", step_metric="Episode")
        wandb.define_metric("Value Loss", step_metric="Episode")
        wandb.define_metric("Total Reward", step_metric="Episode")
        wandb.define_metric("Total Bill", step_metric="Episode")
        wandb.define_metric("Total Sell", step_metric="Episode")
        wandb.define_metric("Total Purchase", step_metric="Episode")
        wandb.define_metric("SoC", step_metric="Episode")
    
    def _build_policy_network(self):
        """Build policy network that outputs action probabilities"""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(self.action_size, activation='softmax')
        ])
        return model
    
    def _build_value_network(self):
        """Build value network that estimates state values"""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1)
        ])
        return model
    
    def get_action_probs(self, state, feasible_actions):
        """Get action probabilities from policy network"""
        with tf.device('/GPU:0'):
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1))
            action_probs = self.policy_network(state_tensor, training=False).numpy()[0]
        
        # Mask infeasible actions
        mask = np.zeros(self.action_size)
        mask[feasible_actions] = 1
        masked_probs = action_probs * mask
        
        # Normalize probabilities
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            masked_probs[feasible_actions] = 1.0 / len(feasible_actions)
        
        return masked_probs
    
    def act(self, state, feasible_actions):
        """Choose action based on policy network"""
        action_probs = self.get_action_probs(state, feasible_actions)
        action = np.random.choice(self.action_size, p=action_probs)
        return action, action_probs[action]
    
    def compute_advantages(self, rewards, values, next_values, dones):
        """Compute advantages using Generalized Advantage Estimation (GAE)"""
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * gae
            advantages[t] = gae
            
        returns = advantages + values
        return advantages, returns
    
    def train_policy(self, states, actions, old_probs, advantages):
        """Train policy network using PPO loss"""
        with tf.GradientTape() as tape:
            action_probs = self.policy_network(states, training=True)
            
            # Get probabilities of taken actions
            indices = tf.range(len(actions))
            action_indices = tf.stack([indices, actions], axis=1)
            current_probs = tf.gather_nd(action_probs, action_indices)
            
            # Calculate ratio and clipped ratio
            ratio = current_probs / old_probs
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            
            # Calculate policy loss
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            
            # Add entropy bonus
            entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1))
            total_loss = policy_loss - self.entropy_beta * entropy
        
        gradients = tape.gradient(total_loss, self.policy_network.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))
        
        return policy_loss.numpy()
    
    def train_value(self, states, returns):
        """Train value network using Huber loss"""
        with tf.GradientTape() as tape:
            values = self.value_network(states, training=True)
            value_loss = Huber(delta=1.0)(returns, tf.squeeze(values))
        
        gradients = tape.gradient(value_loss, self.value_network.trainable_variables)
        self.value_optimizer.apply_gradients(zip(gradients, self.value_network.trainable_variables))
        
        return value_loss.numpy()
    
    def train_on_batch(self, trajectory):
        """Train both networks on a batch of experiences"""
        states = np.array([item[0] for item in trajectory])
        actions = np.array([item[1] for item in trajectory])
        old_probs = np.array([item[2] for item in trajectory])
        rewards = np.array([item[3] for item in trajectory])
        next_states = np.array([item[4] for item in trajectory])
        dones = np.array([item[5] for item in trajectory])
        
        # Get values for current and next states
        with tf.device('/GPU:0'):
            values = tf.squeeze(self.value_network(states)).numpy()
            next_values = tf.squeeze(self.value_network(next_states)).numpy()
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, next_values, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_probs_tensor = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # Train networks multiple times
        policy_losses = []
        value_losses = []
        
        for _ in range(10):  # Number of training iterations
            policy_loss = self.train_policy(states_tensor, actions_tensor, old_probs_tensor, advantages_tensor)
            value_loss = self.train_value(states_tensor, returns_tensor)
            
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
        
        return np.mean(policy_losses), np.mean(value_losses)
    
    def save_checkpoint(self, episode, metrics):
        """Save model checkpoints and log metrics"""
        # Save latest models (overwriting previous)
        policy_path = self.checkpoint_dir / "latest_policy.h5"
        value_path = self.checkpoint_dir / "latest_value.h5"
        self.policy_network.save(policy_path)
        self.value_network.save(value_path)
        
        # Save best models if current reward is better
        if metrics["Total Reward"] > self.best_reward:
            self.best_reward = metrics["Total Reward"]
            best_policy_path = self.checkpoint_dir / "best_policy.h5"
            best_value_path = self.checkpoint_dir / "best_value.h5"
            self.policy_network.save(best_policy_path)
            self.value_network.save(best_value_path)
        
        # Log metrics
        wandb.log({
            "Episode": metrics["Episode"],
            "Policy Loss": metrics["Policy Loss"],
            "Value Loss": metrics["Value Loss"],
            "Total Reward": metrics["Total Reward"],
            "Total Bill": metrics["Total Bill"],
            "Total Sell": metrics["Total Sell"],
            "Total Purchase": metrics["Total Purchase"],
            "SoC": metrics["SoC"]
        }, step=episode)
    
    def load_checkpoint(self, policy_path, value_path):
        """Load models from checkpoints"""
        with tf.device('/GPU:0'):
            self.policy_network = models.load_model(policy_path)
            self.value_network = models.load_model(value_path)
        print(f"Loaded checkpoints from {policy_path} and {value_path}")