import wandb
import numpy as np
from tqdm import tqdm

def train_dqn(env, agent, episodes):
    try:
        log_frequency = 10
        episode_bar = tqdm(range(episodes), desc="Episodes", position=0)
        
        for episode in episode_bar:
            state = env.reset()
            total_reward = 0
            losses = []
            soc_values = []
            total_bill = 0
            total_sell = 0
            total_purchase = 0
            step = 0
            
            step_bar = tqdm(total=env.max_steps, desc=f"Steps (Episode {episode+1})", 
                          position=1, leave=False)
            
            while True:
                feasible_actions = env.get_feasible_actions()
                action = agent.act(state, feasible_actions)
                next_state, reward, done, info = env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                
                loss = 0
                if len(agent.memory) > agent.batch_size:
                    loss = agent.replay(step)
                    losses.append(loss)
                
                total_reward += reward
                soc_values.append(info['SoC'])
                total_bill += info['Bill']
                total_sell += info['Sell']
                total_purchase += info['Purchase']
                
                if step % log_frequency == 0:
                    wandb.log({
                        "Episode": episode,
                        "Step": step,
                        "Step Loss": loss if loss else 0,
                        "Step Reward": reward,
                        "Step Bill": info['Bill'],
                        "Step Sell": info['Sell'],
                        "Step Purchase": info['Purchase'],
                        "Step SoC": info['SoC'],
                        "Running Total Reward": total_reward,
                        "Epsilon": agent.epsilon
                    }, step=episode*env.max_steps + step)
                
                if done:
                    if agent.epsilon > agent.epsilon_min:
                        agent.epsilon *= agent.epsilon_decay
                    
                    avg_loss = np.mean(losses) if losses else 0
                    avg_soc = np.mean(soc_values)
                    
                    metrics = {
                        "Episode": episode,
                        "Loss": avg_loss,
                        "Total Reward": total_reward,
                        "Total Bill": total_bill,
                        "Total Sell": total_sell,
                        "Total Purchase": total_purchase,
                        "Epsilon": agent.epsilon,
                        "SoC": avg_soc
                    }
                    
                    wandb.log(metrics, step=episode)
                    agent.save_checkpoint(episode, metrics)
                    
                    episode_bar.set_postfix({
                        'Reward': f'{total_reward:.2f}',
                        'Epsilon': f'{agent.epsilon:.4f}',
                        'Avg SoC': f'{avg_soc:.2f}%',
                        'Bill': f'${total_bill:.2f}'
                    })
                    
                    step_bar.close()
                    break
                
                state = next_state
                step += 1
                
                step_bar.update(1)
                step_bar.set_postfix({
                    'Reward': f'{total_reward:.2f}',
                    'SoC': f'{info["SoC"]:.2f}%',
                    'Bill': f'${total_bill:.2f}'
                })
                
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
        agent.save_checkpoint(episode, metrics)
        
    finally:
        wandb.finish()

def train_ppo(env, agent, episodes):
    try:
        log_frequency = 10
        trajectory_length = 256  # Length of trajectory before updating
        episode_bar = tqdm(range(episodes), desc="Episodes", position=0)
        
        for episode in episode_bar:
            state = env.reset()
            total_reward = 0
            policy_losses = []
            value_losses = []
            soc_values = []
            total_bill = 0
            total_sell = 0
            total_purchase = 0
            trajectory = []
            step = 0
            
            step_bar = tqdm(total=env.max_steps, desc=f"Steps (Episode {episode+1})", 
                          position=1, leave=False)
            
            while True:
                feasible_actions = env.get_feasible_actions()
                action, action_prob = agent.act(state, feasible_actions)
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                trajectory.append((state, action, action_prob, reward, next_state, done))
                
                # Update metrics
                total_reward += reward
                soc_values.append(info['SoC'])
                total_bill += info['Bill']
                total_sell += info['Sell']
                total_purchase += info['Purchase']
                
                # Train if trajectory is long enough or episode is done
                if len(trajectory) >= trajectory_length or done:
                    policy_loss, value_loss = agent.train_on_batch(trajectory)
                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                    trajectory = []
                
                if step % log_frequency == 0:
                    wandb.log({
                        "Episode": episode,
                        "Step": step,
                        "Step Policy Loss": policy_loss if policy_losses else 0,
                        "Step Value Loss": value_loss if value_losses else 0,
                        "Step Reward": reward,
                        "Step Bill": info['Bill'],
                        "Step Sell": info['Sell'],
                        "Step Purchase": info['Purchase'],
                        "Step SoC": info['SoC'],
                        "Running Total Reward": total_reward
                    }, step=episode*env.max_steps + step)
                
                if done:
                    avg_policy_loss = np.mean(policy_losses) if policy_losses else 0
                    avg_value_loss = np.mean(value_losses) if value_losses else 0
                    avg_soc = np.mean(soc_values)
                    
                    metrics = {
                        "Episode": episode,
                        "Policy Loss": avg_policy_loss,
                        "Value Loss": avg_value_loss,
                        "Total Reward": total_reward,
                        "Total Bill": total_bill,
                        "Total Sell": total_sell,
                        "Total Purchase": total_purchase,
                        "SoC": avg_soc
                    }
                    
                    wandb.log(metrics, step=episode)
                    agent.save_checkpoint(episode, metrics)
                    
                    episode_bar.set_postfix({
                        'Reward': f'{total_reward:.2f}',
                        'P_Loss': f'{avg_policy_loss:.4f}',
                        'V_Loss': f'{avg_value_loss:.4f}',
                        'Avg SoC': f'{avg_soc:.2f}%',
                        'Bill': f'${total_bill:.2f}'
                    })
                    
                    step_bar.close()
                    break
                
                state = next_state
                step += 1
                
                step_bar.update(1)
                step_bar.set_postfix({
                    'Reward': f'{total_reward:.2f}',
                    'SoC': f'{info["SoC"]:.2f}%',
                    'Bill': f'${total_bill:.2f}'
                })
                
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
        agent.save_checkpoint(episode, metrics)
        
    finally:
        wandb.finish()