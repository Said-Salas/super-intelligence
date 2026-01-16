import numpy as np
import math
from rl_interface import GenesisEnvironment

class TreadmillEnvironment(GenesisEnvironment):
    """
    A simple physics simulation to test the PPO brain.
    Task: Move a 1D agent to a target position.
    State: [position, velocity, target_position]
    Action: [force]
    """
    def __init__(self):
        self._action_size = 1
        self._obs_size = 3
        
        # Physics State
        self.pos = 0.0
        self.vel = 0.0
        self.target = 0.0
        self.max_steps = 200
        self.current_step = 0
        
    @property
    def action_space_size(self) -> int:
        return self._action_size

    @property
    def observation_space_size(self) -> int:
        return self._obs_size

    def reset(self) -> np.ndarray:
        self.pos = 0.0
        self.vel = 0.0
        # Random target between -5 and 5
        self.target = np.random.uniform(-5.0, 5.0)
        self.current_step = 0
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple:
        # 1. Apply Physics
        force = np.clip(action[0], -1.0, 1.0) * 0.5 # Clamp and scale force
        
        # Simple drag
        self.vel *= 0.95
        
        self.vel += force
        self.pos += self.vel
        
        # 2. Calculate Reward
        # Distance to target
        dist = abs(self.pos - self.target)
        
        # Reward: Higher is better. 
        # 1.0 / (dist + 0.1) gives a spike near target.
        # Subtracting small penalty for effort/time to encourage speed.
        reward = 1.0 / (dist + 0.1) - 0.01 * abs(force)
        
        # 3. Check Termination
        self.current_step += 1
        done = False
        
        # Success condition
        if dist < 0.1:
            reward += 10.0 # Big bonus for hitting target
            done = True
            
        # Timeout condition
        if self.current_step >= self.max_steps:
            done = True
            
        # Fail condition (wandering too far)
        if abs(self.pos) > 20.0:
            reward -= 10.0
            done = True
            
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.array([self.pos, self.vel, self.target], dtype=np.float32)

    def close(self):
        pass

