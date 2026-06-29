from copy import deepcopy
from duckietown_utils.env import launch_and_wrap_env
from duckietown_utils.wrappers.simulator_mod_wrappers import ObstacleSpawningWrapper
from duckietown_utils.rllib_callbacks import on_train_result as orig_on_train_result

# Thresholds are env-specific — tune these empirically.
# None on the last stage means "never auto-advance".
CURRICULUM = [
    {'density': 0.0, 'static': True,  'spawn': True, 'min_reward': 100,  'label': 'No obstacles'},
    {'density': 0.2, 'static': True,  'spawn': True,  'min_reward': 100,  'label': 'Static obstacles'},
    {'density': 0.3, 'static': False, 'spawn': True,  'min_reward': 100,  'label': 'Moving obstacles (sparse)'},
    {'density': 0.5, 'static': False, 'spawn': True,  'min_reward': None,'label': 'Full difficulty'},
]
STAGE_STABILITY = 4  # consecutive iters above threshold before advancin

###########################################################
# Custom Train Result Callback for Curriculum Advancement
###########################################################
def custom_on_train_result(info):
    """
    Evaluates the mean reward and advances the curriculum stage across all Ray workers.
    """
    # 1. Call the original callback to maintain histogram/stats clearing logi
    orig_on_train_result(info)
    trainer = info["trainer"]
    result = info["result"]
    
    # Initialize curriculum state on the trainer if not present
    if not hasattr(trainer, "curriculum_stage_idx"):
        # Pull the stage from the config we passed down, rather than defaulting to 0
        trainer.curriculum_stage_idx = trainer.config["env_config"].get("resume_stage", 0)
        trainer.iters_at_stage = 0
        
    stage_idx = trainer.curriculum_stage_idx

    result["curriculum_stage"] = stage_idx
    
    # If we've reached the last stage, do nothing
    if stage_idx >= len(CURRICULUM) - 1:
        return
        
    stage = CURRICULUM[stage_idx]
    mean_rwd = result.get("episode_reward_mean", 0.0)
    threshold = stage["min_reward"]
    
    if threshold is not None and mean_rwd >= threshold:
        trainer.iters_at_stage += 1
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"  [Curriculum] Above threshold ({mean_rwd:.1f} >= {threshold}) for {trainer.iters_at_stage}/{STAGE_STABILITY} iters.")
        
        if trainer.iters_at_stage >= STAGE_STABILITY:
            trainer.curriculum_stage_idx += 1
            trainer.iters_at_stage = 0
            next_stage = CURRICULUM[trainer.curriculum_stage_idx]
            logger.info(f"  >>> ADVANCING to stage {trainer.curriculum_stage_idx}: {next_stage['label']}")
            
            # Define the function to update the environment in-place
            def set_stage_on_env(env):
                from duckietown_utils.wrappers.simulator_mod_wrappers import ObstacleSpawningWrapper
                curr = env
                while hasattr(curr, 'env'):
                    if isinstance(curr, ObstacleSpawningWrapper):
                        curr.env_config['spawn_obstacles'] = next_stage['spawn']
                        if 'obstacles' not in curr.env_config:
                            curr.env_config['obstacles'] = {'duckie': {}}
                        curr.env_config['obstacles']['duckie']['density'] = next_stage['density']
                        curr.env_config['obstacles']['duckie']['static'] = next_stage['static']
                        break
                    curr = curr.env
                    
            # Broadcast the update to ALL remote workers and environments
            trainer.workers.foreach_worker(
                lambda worker: worker.foreach_env(set_stage_on_env)
            )
            
            # Update the result dictionary so the jump is immediately logged on this iteration
            result["curriculum_stage"] = trainer.curriculum_stage_idx
    else:
        # Reset stability counter if it dips below threshold
        trainer.iters_at_stage = 0


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