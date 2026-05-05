from src.gym_duckietown.simulator import Simulator
from src.gym_duckietown.render import render_top_down_map
from src.gym_duckietown.hierarchical_wrapper import HierarchicalDuckietownWrapper
import cv2
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_name", type=str, default="straight_road", choices=["straight_road", "4way", "udem1", "small_loop", "small_loop_cw", "zigzag_dists", "loop_obstacles", "loop_pedestrians"])
    args = parser.parse_args()

    base_env = Simulator(
        seed=123, 
        map_name=args.map_name, 
        max_steps=500001,
        camera_width=640,
        camera_height=480
    )
    
    env = HierarchicalDuckietownWrapper(base_env)
    obs = env.reset()
    
    raw_map_img = render_top_down_map(env)
    
    while True:
        action = np.array([0.4, 0.2]) 
        
        obs, reward, done, misc = env.step(action)
        
        img_bgr = cv2.cvtColor(env.render("rgb_array"), cv2.COLOR_RGB2BGR)
        cv2.imshow("Duckiebot Camera", img_bgr)
        
        cv2.imshow("Original Map", raw_map_img)
        
        path_map_img = render_top_down_map(env, path=env.current_path, current_pos=env.unwrapped.cur_pos)
        cv2.imshow("A* Global Planner", path_map_img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if done:
            env.reset()

    env.close()
    cv2.destroyAllWindows()