import gymnasium as gym
import numpy as np
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, MaxAndSkipObservation

class HardTrackEnv(gym.Wrapper):
    def __init__(self, render_mode="rgb_array"):
        env = gym.make(
            'CarRacing-v3',
            domain_randomize=False, # No random seeds
            render_mode=render_mode,
            continuous=False, # Discrete actions
            max_episode_steps=2000
        )
        
        # preprocessing
        env = MaxAndSkipObservation(env, skip=4)
        env = GrayscaleObservation(env, keep_dim=False)
        env = ResizeObservation(env, shape=(84, 84))
        env = FrameStackObservation(env, stack_size=4)
        
        super().__init__(env)
        # Hard track seed
        self.seed = 9876
        
    def reset(self):
        return self.env.reset(seed=self.seed)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if reward > 0:
            reward = reward * 0.8 # harsher for hard track
        elif reward < 0:
            reward = reward * 1.2
        
        return obs, reward, terminated, truncated, info

# Test if file works
if __name__ == "__main__":
    env = HardTrackEnv(render_mode="human")
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    done = False
    while not done:
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        done = terminated or truncated
        
        if steps % 10 == 0:
            print(f"Steps: {steps}, Total Reward: {total_reward:.2f}")
        
        if steps >= 1000:
            break
    
    print(f"Test finished, Steps: {steps}, Final Reward: {total_reward:.2f}")
    env.close()