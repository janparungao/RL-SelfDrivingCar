import torch
import gymnasium as gym
import numpy as np
import time
import os
import random
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, MaxAndSkipObservation
from DQNAgent import DQNAgent

MODEL_PATH = './models/hard_best_model.pt'  # easy or hard
NUM_EPISODES = 10                           
MAX_TIMESTEPS = 2000                        
RENDER = True # True for visual, false for no visual

def evaluate_standard_gym():
    env = gym.make('CarRacing-v3', 
                   render_mode="human" if RENDER else "rgb_array", 
                  continuous=False, 
                  domain_randomize=False)
    
    env = MaxAndSkipObservation(env, skip=4)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStackObservation(env, stack_size=4)
    
    # random seeds
    seeds = [42, 123, 7, 999, 2023, 8675, 309, 1337, 4242, 5050]
    
    obs, _ = env.reset(seed=seeds[0])
    input_shape = obs.shape
    n_actions = env.action_space.n
    agent = DQNAgent(input_shape, n_actions)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=agent.device)
        agent.q_network.load_state_dict(checkpoint['q_network'])
        agent.target_network.load_state_dict(checkpoint['target_network'])
        print(f"Successfully loaded model from {MODEL_PATH}")
        if 'episode' in checkpoint:
            print(f"This model was saved after episode {checkpoint['episode']}")
    except Exception as e:
        print(f"No car found: {e}")
        return
    
    agent.epsilon = 0.0
    scores = []
    timestep_arr = []
    lap_times = []
    completion_count = 0
    
    for episode in range(NUM_EPISODES):
        seed_idx = episode % len(seeds)
        current_seed = seeds[seed_idx]
        state, _ = env.reset(seed=current_seed)
        print(f"Episode {episode+1}/{NUM_EPISODES} - Using seed: {current_seed}")
        
        starting_position = None
        try:
            if hasattr(env.unwrapped, 'car'):
                starting_position = (env.unwrapped.car.hull.position.x, env.unwrapped.car.hull.position.y)
        except:
            print("Warning: Could not access car position")
        
        score = 0
        timestep = 0
        lap_complete = False
        
        while timestep < MAX_TIMESTEPS:
            with torch.no_grad():
                state_tensor = torch.tensor(state).float().to(agent.device)
                state_tensor = state_tensor.unsqueeze(0)
                q_values = agent.q_network(state_tensor)
                action = q_values.max(1)[1].view(1, 1)
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            state = next_state
            score += reward
            timestep += 1
            
            if not lap_complete and starting_position is not None and timestep > 50:
                try:
                    current_pos = None
                    if hasattr(env.unwrapped, 'car'):
                        current_pos = (env.unwrapped.car.hull.position.x, env.unwrapped.car.hull.position.y)
                    
                    if current_pos is not None:
                        distance_to_start = ((current_pos[0] - starting_position[0])**2 + 
                                           (current_pos[1] - starting_position[1])**2)**0.5
                        
                        if distance_to_start < 10.0:
                            lap_complete = True
                            lap_times.append(timestep)
                            completion_count += 1
                            print(f"  Lap completed in {timestep} timesteps!")
                except:
                    pass
            
            if done:
                break
        
        scores.append(score)
        timestep_arr.append(timestep)
        
        print(f"Final score: {score:.1f}, Total timesteps: {timestep}")
        if not lap_complete:
            print("Did not complete lap")
    
    if lap_times:
        avg_lap_time = np.mean(lap_times)
        std_lap_time = np.std(lap_times)
        print(f"Average lap time: {avg_lap_time:} ± {std_lap_time:.} timesteps")
        print(f"Best lap time: {min(lap_times)} timesteps")
    else:
        print("No laps were completed during evaluation.")
        
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    avg_timesteps = np.mean(timestep_arr)
    std_timesteps = np.std(timestep_arr)
    
    print(f"Avg score: {avg_score:.1f} (±{std_score:.1f})")
    print(f"Avg timesteps: {avg_timesteps:} (±{std_timesteps:}")
    print(f"Min score: {min(scores):}, Max score: {max(scores):}")
    print(f"Min timesteps: {min(timestep_arr)}, Max timesteps: {max(timestep_arr)}")
    print(f"Finished {completion_count} out of {NUM_EPISODES} episodes ({completion_count/NUM_EPISODES*100:}%)")
    
    env.close()
    
    return scores, timestep_arr, lap_times

if __name__ == "__main__":
    evaluate_standard_gym()