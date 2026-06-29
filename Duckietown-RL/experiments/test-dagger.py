"""
Script to evaluate the DAgger-trained CNN agent.
To select a trained agent use the --seed-model-id argument (e.g. --seed-model-id 3045).
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024"

import time
import os
import logging
import numpy as np
import argparse
import cv2
import tensorflow as tf

from config.config import find_and_load_config_by_seed, update_config
from duckietown_utils.env import launch_and_wrap_env
from duckietown_utils.utils import seed
from duckietown_utils.duckietown_world_evaluator import DuckietownWorldEvaluator, DEFAULT_EVALUATION_MAP
from duckietown_utils.rllib_callbacks import *

from config.curriculum import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = ''

class KerasAgentWrapper:
    """
    A simple wrapper to make a Keras model act like an RLlib trainer for evaluation.
    DuckietownWorldEvaluator expects an agent with a compute_action method.
    """
    def __init__(self, model):
        self.model = model

    def compute_action(self, obs, explore=False):
        # Using model() instead of model.predict() is much faster for single samples
        obs_batch = np.expand_dims(obs, axis=0)
        obs_batch = obs_batch.astype(np.float32)
        action = self.model.predict(obs_batch)[0]
        return action

if __name__ == "__main__":
    ###########################################################
    # Read and process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed-model-id', default=42, type=int,
                        help='Unique experiment identifier for the config.')
    parser.add_argument('--analyse-trajectories', action='store_true',
                        help='Calculate metrics and create trajectory plots.')
    parser.add_argument('--map-name', default=DEFAULT_EVALUATION_MAP, help="Specify the map")
    parser.add_argument('--top-view', action='store_true',
                        help="View the simulation from a fixed bird's eye view, instead of the robot's view")

    parser.add_argument('--curriculum-base', dest='curriculum_base', action='store_true', default=True,
                    help='Use curriculum learning (default).')
    parser.add_argument('--no-curriculum-base', dest='curriculum_base', action='store_false',
                    help='Disable curriculum learning.')
    parser.add_argument('--curriculum', dest='curriculum', action='store_true', default=True,
                    help='Use curriculum learning (default).')
    parser.add_argument('--no-curriculum', dest='curriculum', action='store_false',
                    help='Disable curriculum learning.')
    parser.add_argument('-e', '--experiment-idx', default=0, type=int,
                    help='Experiment Id')
    parser.add_argument('-c', '--checkpoint-idx', default=0, type=int,
                    help='Checkpoint Id')
    parser.add_argument('-m', '--model-path', default='', type=str,
                    help='Model Path')
    parser.add_argument('--grayscale', dest='grayscale', default=True, action='store_true',
                    help='Grayscale')
    parser.add_argument('--no-grayscale', dest='grayscale', action='store_false',
                    help='No Grayscale')
    args = parser.parse_args()

    if args.top_view:
        render_mode = 'top_down'
    else:
        render_mode = 'human'

    test_map = args.map_name

    seed(1234)
    SEED = args.seed_model_id
    filt = None
    if args.model_path != '':
        model_path = args.model_path
        results_path = 'eval_results_' + model_path.split('.h5')[0]
    elif args.curriculum_base:
        filt = 'Curriculum'
        if args.curriculum:
            results_path = f"artifacts/eval_results_dagger_seed_{SEED}_cur_both"
            model_path = f"artifacts/dagger_model_seed_{SEED}_cur_both_best.h5"
        else:
            results_path = f"artifacts/eval_results_dagger_seed_{SEED}_cur_ppo"
            model_path = f"artifacts/dagger_model_seed_{SEED}_cur_ppo_best.h5"
    else:
        if args.curriculum:
           results_path = f"artifacts/eval_results_dagger_seed_{SEED}_cur_dag" 
           model_path = f"artifacts/dagger_model_seed_{SEED}_cur_dag_best.h5"
        else:
            results_path = f"artifacts/eval_results_dagger_seed_{SEED}"
            model_path = f"artifacts/dagger_model_seed_{SEED}_best.h5"
    ###########################################################
    # Load experiment config (Used to initialize the environment perfectly)
    SEED = args.seed_model_id
    config, _ = find_and_load_config_by_seed(SEED, preselected_experiment_idx=args.experiment_idx, preselected_checkpoint_idx=args.checkpoint_idx, experiment_name_filter=filt)
    
    update_config(config, {
        'env_config': {
            'mode': 'inference',
            'training_map': DEFAULT_EVALUATION_MAP,
            "domain_rand": True,
            "dynamics_rand": True,
            "camera_rand": True,
            "grayscale_image":args.grayscale,
            "spawn_obstacles": True,
            "spawn_forward_obstacle": False,
            "obstacles": {
                "duckie": {
                    "density": 0.5,
                    "static": False,
                }
            }
        }
    })

    ###########################################################
    # Load Novice Agent (Keras Model)
    print(f"\n>>> Loading DAgger model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Have you run train-dagger.py yet?")
        
    novice_model = tf.keras.models.load_model(model_path)
    agent = KerasAgentWrapper(novice_model)

    ###########################################################
    # Simple demonstration of closed loop performance
    if not args.analyse_trajectories:
        env = launch_and_wrap_env(config["env_config"])
        for i in range(5):
            obs = env.reset()
            env.render(render_mode)
            done = False
            while not done:
                action = agent.compute_action(obs, explore=False)
                obs, reward, done, info = env.step(action)
                
                # Show observation camera
                cv2.imshow("Observation", cv2.cvtColor(cv2.resize(obs[..., -3:].astype('float32'), (300, 300)), cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
                
                # Render simulator window
                orig_distortion = env.unwrapped.distortion
                env.unwrapped.distortion = False
                env.render(render_mode)
                env.unwrapped.distortion = orig_distortion
                
                if env.unwrapped.frame_skip > 1:
                    time.sleep(env.unwrapped.delta_time * env.unwrapped.frame_skip)
                    
        env.close()

    ###########################################################
    
    config['env_config']['spawn_forward_obstacle'] = False 
    
    evaluator = DuckietownWorldEvaluator(config['env_config'], eval_lenght_sec=15, eval_map=test_map)
    
    print(f"\n>>> Running Trajectory Analysis. Results will be saved to {results_path}...")
    
    evaluator.evaluate(agent, results_path, episodes=25)
