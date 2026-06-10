from copy import deepcopy
from duckietown_utils.env import launch_and_wrap_env
from duckietown_utils.wrappers.simulator_mod_wrappers import ObstacleSpawningWrapper

# Thresholds are env-specific — tune these empirically.
# None on the last stage means "never auto-advance".
CURRICULUM = [
    {'density': 0.0, 'static': True,  'spawn': True, 'min_reward': 100,  'label': 'No obstacles'},
    {'density': 0.2, 'static': True,  'spawn': True,  'min_reward': 100,  'label': 'Static obstacles'},
    {'density': 0.3, 'static': False, 'spawn': True,  'min_reward': 100,  'label': 'Moving obstacles (sparse)'},
    {'density': 0.5, 'static': False, 'spawn': True,  'min_reward': None,'label': 'Full difficulty'},
]
STAGE_STABILITY = 2  # consecutive iters above threshold before advancin

def advance_stage(env, stage):
    """
    Updates the curriculum stage in-place without destroying the Pyglet window.
    This prevents OpenGL memory leaks.
    """
    curr = env
    wrapper_found = False
    
    # Traverse the Gym wrapper stack to find the ObstacleSpawningWrapper
    while hasattr(curr, 'env'):
        if isinstance(curr, ObstacleSpawningWrapper):
            # Update the configuration dictionary in place
            curr.env_config['spawn_obstacles'] = stage['spawn']
            if 'obstacles' not in curr.env_config:
                curr.env_config['obstacles'] = {'duckie': {}}
            
            curr.env_config['obstacles']['duckie']['density'] = stage['density']
            curr.env_config['obstacles']['duckie']['static']  = stage['static']
            wrapper_found = True
            break
        curr = curr.env
        
    return env