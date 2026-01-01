import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
from DQNAgent import DQNAgent
from easy_trackenv import EasyTrackEnv
from hard_trackenv import HardTrackEnv


ENV_TYPE = 'hard' # 'easy' or 'hard'
NUM_EPISODES = 500
SAVE_DIR = './models'

def train():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    env = EasyTrackEnv(render_mode="rgb_array") if ENV_TYPE == 'easy' else HardTrackEnv(render_mode="rgb_array")
    
    obs, _ = env.reset()
    agent = DQNAgent(obs.shape, env.action_space.n)
    
    rewards = []
    best_reward = -float('inf')
    recent_rewards = deque(maxlen=100)
    
    # Training loop
    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            next_state_tensor = None if done else torch.tensor(next_state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=agent.device)
            agent.memory.push(state_tensor, action, next_state_tensor, reward_tensor, done)
            
            state = next_state
            episode_reward += reward
            agent.learn()
        
        rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        avg_reward = np.mean(recent_rewards) if recent_rewards else episode_reward
        
        if episode % agent.target_update_freq == 0:
            agent.update_target()
            print(f"Episode {episode}: target network updated")
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{NUM_EPISODES}, Reward: {episode_reward:.1f}, Avg: {avg_reward:.1f}")
        
        if avg_reward > best_reward and len(recent_rewards) == 100:
            best_reward = avg_reward
            torch.save({
                'q_network': agent.q_network.state_dict(),
                'target_network': agent.target_network.state_dict(),
                'epsilon': agent.epsilon,
                'episode': episode
            }, f"{SAVE_DIR}/{ENV_TYPE}_best_model.pt")
        
        if episode % 100 == 0:
            torch.save({
                'q_network': agent.q_network.state_dict(),
                'target_network': agent.target_network.state_dict(),
                'epsilon': agent.epsilon,
                'episode': episode
            }, f"{SAVE_DIR}/{ENV_TYPE}_checkpoint_ep{episode}.pt")
    
    # save final model
    torch.save({
        'q_network': agent.q_network.state_dict(),
        'target_network': agent.target_network.state_dict(),
        'epsilon': agent.epsilon,
        'episode': NUM_EPISODES
    }, f"{SAVE_DIR}/{ENV_TYPE}_final_model.pt")
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Reward')
    if len(rewards) >= 100:
        avg_rewards = [np.mean(rewards[i:i+100]) for i in range(len(rewards) - 100 + 1)]
        plt.plot(range(100-1, len(rewards)), avg_rewards, label='Average over 100 episodes')
        
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Rewards on {ENV_TYPE} Track')
    plt.legend()
    plt.savefig(f'{ENV_TYPE}_rewards.png')
    plt.show()
    
    return agent, rewards

if __name__ == "__main__":
    train()