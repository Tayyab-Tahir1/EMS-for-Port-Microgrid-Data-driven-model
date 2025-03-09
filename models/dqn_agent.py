import os
import wandb
import tensorflow as tf
import numpy as np
import random
from collections import deque
from tensorflow.keras import models, layers, optimizers
from pathlib import Path

class DQNAgent:
    def __init__(
        self, 
        state_size, 
        action_size, 
        initial_epsilon=1.0, 
        epsilon_min=0.01,
        episodes=200,
        case_name="default"
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.90
        self.epsilon = initial_epsilon
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.learning_rate = 1e-5
        self.batch_size = 128
        self.update_target_frequency = 50
        self.case_name = case_name
        self.best_reward = float('-inf')
        
        self.checkpoint_dir = Path(f'checkpoints/{case_name}')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Calculate epsilon decay rate
        self.epsilon_decay = self.calculate_epsilon_decay()
        
        # Set XLA GPU flags
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/apps/cuda/11.2.2'
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
        
        with tf.device('/GPU:0'):
            self.model = self._build_model()
            self.target_model = self._build_model()
        self.update_target_network()
        
        self.init_wandb()

    def calculate_epsilon_decay(self):
        """Calculate epsilon decay rate to reach epsilon_min after given episodes"""
        return np.exp(np.log(self.epsilon_min/self.epsilon) / self.episodes)
        
    def init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project="Energy Management System",
            name=f"EMS-Port-{self.case_name}",
            config={
                "learning_rate": self.learning_rate,
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "gamma": self.gamma,
                "batch_size": self.batch_size,
                "memory_size": self.memory.maxlen,
                "case_name": self.case_name
            }
        )
        
        # Configure metrics
        wandb.define_metric("Episode")
        wandb.define_metric("Loss", step_metric="Episode")
        wandb.define_metric("Total Reward", step_metric="Episode")
        wandb.define_metric("Total Bill", step_metric="Episode")
        wandb.define_metric("Total Sell", step_metric="Episode")
        wandb.define_metric("Total Purchase", step_metric="Episode")
        wandb.define_metric("Epsilon", step_metric="Episode")
        wandb.define_metric("SoC", step_metric="Episode")
        
    def _build_model(self):
        """Build neural network model"""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        opt = optimizers.Adam(
            learning_rate=self.learning_rate,
            clipnorm=0.1,
            clipvalue=0.5
        )
        
        model.compile(
            loss=tf.keras.losses.Huber(delta=0.5),
            optimizer=opt,
            metrics=['mse'],
            run_eagerly=False,
            jit_compile=False
        )
        return model
            
    def update_target_network(self):
        """Update target network with weights from main network"""
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, feasible_actions):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.choice(feasible_actions)
        
        with tf.device('/GPU:0'):
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1))
            act_values = self.model(state_tensor, training=False).numpy()
        
        mask = np.full(self.action_size, -np.inf)
        mask[feasible_actions] = 0
        masked_act_values = act_values[0] + mask
        
        return np.argmax(masked_act_values)

    def replay(self, step):
        """Train on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0
            
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([item[0] for item in minibatch])
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        next_states = np.array([item[3] for item in minibatch])
        dones = np.array([item[4] for item in minibatch])
        
        with tf.device('/GPU:0'):
            # Convert to tensors
            states_tensor = tf.convert_to_tensor(states)
            next_states_tensor = tf.convert_to_tensor(next_states)
            
            # Get predictions
            current_q_values = self.model(states_tensor, training=False).numpy()
            target_next_q_values = self.target_model(next_states_tensor, training=False).numpy()
            
            # Create targets
            targets = current_q_values.copy()
            for i in range(self.batch_size):
                if dones[i]:
                    targets[i, actions[i]] = rewards[i]
                else:
                    max_next_q = np.max(target_next_q_values[i])
                    targets[i, actions[i]] = rewards[i] + self.gamma * max_next_q
            
            # Train the model
            history = self.model.fit(
                states, 
                targets, 
                batch_size=self.batch_size,
                epochs=1, 
                verbose=0
            )
            
            if step % self.update_target_frequency == 0:
                self.update_target_network()
                
            return history.history['loss'][0]
            
    def save_checkpoint(self, episode, metrics):
        """Save model checkpoints and log metrics"""
        # Save latest model (overwriting previous)
        checkpoint_path = self.checkpoint_dir / "latest_model.h5"
        self.model.save(checkpoint_path)
        
        # Save best model if current reward is better
        if metrics["Total Reward"] > self.best_reward:
            self.best_reward = metrics["Total Reward"]
            best_model_path = self.checkpoint_dir / "best_model.h5"
            self.model.save(best_model_path)
        
        # Log with consistent step
        wandb.log({
            "Episode": metrics["Episode"],
            "Loss": metrics["Loss"],
            "Total Reward": metrics["Total Reward"],
            "Total Bill": metrics["Total Bill"],
            "Total Sell": metrics["Total Sell"],
            "Total Purchase": metrics["Total Purchase"],
            "Epsilon": metrics["Epsilon"],
            "SoC": metrics["SoC"]
        }, step=episode)
        
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        with tf.device('/GPU:0'):
            self.model = models.load_model(checkpoint_path)
            self.target_model.set_weights(self.model.get_weights())
        print(f"Loaded checkpoint from {checkpoint_path}")