import numpy as np
import cv2

def render_top_down_map(env, path=None, current_pos=None):
    scale = 40
    w, h = env.unwrapped.grid_width, env.unwrapped.grid_height
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    
    for i in range(w):
        for j in range(h):
            tile = env.unwrapped._get_tile(i, j)
            color = (50, 50, 50)
            if tile is not None and tile['drivable']:
                color = (200, 200, 200)
            cv2.rectangle(img, (i*scale, j*scale), ((i+1)*scale, (j+1)*scale), color, -1)
            cv2.rectangle(img, (i*scale, j*scale), ((i+1)*scale, (j+1)*scale), (100, 100, 100), 1)

    if path and len(path) > 0:
        for idx in range(len(path) - 1):
            p1 = (int(path[idx][0]*scale + scale/2), int(path[idx][1]*scale + scale/2))
            p2 = (int(path[idx+1][0]*scale + scale/2), int(path[idx+1][1]*scale + scale/2))
            cv2.line(img, p1, p2, (0, 255, 0), 3) # green path of the A* route
            
        goal = path[-1]
        cv2.circle(img, (int(goal[0]*scale + scale/2), int(goal[1]*scale + scale/2)), 8, (0, 0, 255), -1) # red dot for goal

    if current_pos is not None:
        grid_pos = env.get_grid_coords(current_pos)
        center = (int(grid_pos[0]*scale + scale/2), int(grid_pos[1]*scale + scale/2))
        cv2.circle(img, center, 6, (255, 0, 0), -1) # blue dot for robot

    return img