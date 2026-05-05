import gym_duckietown
import gym
from src.gym_duckietown.astar import astar
import numpy as np
from collections import deque
import random

class HierarchicalDuckietownWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.position_history = deque(maxlen=30)
        self.stuck_threshold = 0.05
        self.current_goal_tile = None
        self.current_path = []
        
    def get_grid_coords(self, pos):
        i = int(pos[0] / self.unwrapped.road_tile_size)
        j = int(pos[2] / self.unwrapped.road_tile_size)
        return (i, j)

    def _pick_random_goal_tile(self):
        drivable_tiles = []
        for i in range(self.unwrapped.grid_width):
            for j in range(self.unwrapped.grid_height):
                tile = self.unwrapped._get_tile(i, j)
                if tile is not None and tile['drivable']:
                    drivable_tiles.append((i, j))
        return random.choice(drivable_tiles)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.position_history.clear()
        
        start_tile = self.get_grid_coords(self.unwrapped.cur_pos)
        self.current_goal_tile = self._pick_random_goal_tile()
        
        print(f"[Planner] New Episode! Start: {start_tile} | Goal: {self.current_goal_tile}")
        self.current_path = astar(self.unwrapped, start_tile, self.current_goal_tile)
        
        return obs
        
    def step(self, action):
        obs, reward, done, misc = self.env.step(action)
        current_pos = self.unwrapped.cur_pos
        current_tile = self.get_grid_coords(current_pos)
        
        self.position_history.append(current_pos)
        
        if current_tile == self.current_goal_tile:
            print("[Planner] Reached Goal Tile! Ending Episode.")
            done = True
            reward += 100
            
        if len(self.position_history) == self.position_history.maxlen:
            oldest_pos = self.position_history[0]
            distance_moved = np.linalg.norm(current_pos - oldest_pos)
            
            if distance_moved < self.stuck_threshold:
                print("[Planner] Agent is stuck! Recalculating A* path...")
                self.current_path = astar(self.unwrapped, current_tile, self.current_goal_tile)
                self.position_history.clear() 
                
        return obs, reward, done, misc