"""
Script to evaluate the RLlib PPO agent trained with a YOLO ONNX backbone.
To select a trained agent use the --seed-model-id argument (e.g. --seed-model-id 1118).
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024"

import time
import os
import copy
import logging
import numpy as np
import argparse
import cv2
import gym
import ray
import tensorflow as tf
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer

from config.config import find_and_load_config_by_seed, update_config
from duckietown_utils.env import launch_and_wrap_yolo_env
from duckietown_utils.utils import seed
from duckietown_utils.duckietown_world_evaluator import DuckietownWorldEvaluator, DEFAULT_EVALUATION_MAP
from duckietown_utils.rllib_callbacks import *

from config.curriculum import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = ''

class YoloDuckietownWorldEvaluator(DuckietownWorldEvaluator):
    def __init__(self, env_config, eval_lenght_sec=15, eval_map=DEFAULT_EVALUATION_MAP):
        super().__init__(env_config, eval_lenght_sec, eval_map)
        
        # Override the standard environment with our YOLO wrapper environment
        _env_config = copy.deepcopy(env_config)
        _env_config['episode_max_steps'] = eval_lenght_sec * _env_config['simulation_framerate']
        _env_config['training_map'] = eval_map
        self.env = launch_and_wrap_yolo_env(_env_config)

if __name__ == "__main__":
    ###########################################################
    # Read and process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed-model-id', default=1118, type=int,
                        help='Unique experiment identifier for the config.')
    parser.add_argument('--yolo-onnx-path', type=str, default='artifacts/model_yolo/best.onnx', 
                        help='Path to the ONNX YOLO model')
    parser.add_argument('--analyse-trajectories', action='store_true',
                        help='Calculate metrics and create trajectory plots.')
    parser.add_argument('--map-name', default=DEFAULT_EVALUATION_MAP, help="Specify the map")
    parser.add_argument('--top-view', action='store_true',
                        help="View the simulation from a fixed bird's eye view, instead of the robot's view")

    parser.add_argument('--curriculum', dest='curriculum', action='store_true', default=True,
                    help='Use curriculum learning (default).')
    parser.add_argument('--no-curriculum', dest='curriculum', action='store_false',
                    help='Disable curriculum learning.')
    parser.add_argument('-e', '--experiment-idx', default=0, type=int,
                    help='Experiment Id')
    parser.add_argument('-c', '--checkpoint-idx', default=1, type=int,
                    help='Checkpoint Id')
    args = parser.parse_args()
    args.yolo_onnx_path = os.path.abspath(args.yolo_onnx_path)

    if args.top_view:
        render_mode = 'top_down'
    else:
        render_mode = 'human'

    test_map = args.map_name

    seed(1234)
    SEED = args.seed_model_id
    
    if args.curriculum:
        filt = 'Yolo_CV2_Backbone_PPO_Curriculum'
        results_path = f"artifacts/eval_results_yolo_seed_{SEED}_cur"
    else:
        filt = 'Yolo_CV2_Backbone_PPO'
        results_path = f"artifacts/eval_results_yolo_seed_{SEED}"

    ###########################################################
    # Load experiment config (Used to initialize the environment perfectly)
    print(f"\n>>> Locating RLlib checkpoint for seed {SEED}...")
    config, checkpoint_path = find_and_load_config_by_seed(
        SEED, 
        preselected_experiment_idx=args.experiment_idx, 
        preselected_checkpoint_idx=args.checkpoint_idx, 
        experiment_name_filter=filt
    )
    
    update_config(config, {
        'env_config': {
            'mode': 'inference',
            'training_map': DEFAULT_EVALUATION_MAP,
            "domain_rand": True,
            "dynamics_rand": True,
            "camera_rand": True,
            "grayscale_image": False, # YOLO requires RGB
            "spawn_obstacles": True,
            "spawn_forward_obstacle": False,
            "yolo_onnx_path": args.yolo_onnx_path,
            "obstacles": {
                "duckie": {
                    "density": 0.5,
                    "static": False,
                }
            }
        }
    })

    ###########################################################
    # Initialize Ray and load RLlib Agent
    ray.init(**config.get("ray_init_config", {}))
    register_env('Duckietown', launch_and_wrap_yolo_env)

    print(f">>> Restoring RLlib PPOTrainer from: {checkpoint_path}")
    trainer = PPOTrainer(config=config["rllib_config"])
    trainer.restore(checkpoint_path)

    ###########################################################
    # Simple demonstration of closed loop performance
    if not args.analyse_trajectories:
        env = launch_and_wrap_yolo_env(config["env_config"])
        for i in range(5):
            obs = env.reset()
            env.render(render_mode)
            done = False
            while not done:
                # trainer.compute_action is fully compatible with evaluation
                action = trainer.compute_action(obs, explore=False)
                obs, reward, done, info = env.step(action)
                
                # We can't render the 1D YOLO 'obs', so we pull the raw visual image from the unwrapped env
                raw_rgb = env.unwrapped.render_obs()
                cv2.imshow("Observation Camera", cv2.cvtColor(cv2.resize(raw_rgb, (300, 300)), cv2.COLOR_RGB2BGR))
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
    # Run Trajectory Analysis
    else:
        config['env_config']['spawn_forward_obstacle'] = False 
        
        evaluator = YoloDuckietownWorldEvaluator(config['env_config'], eval_lenght_sec=15, eval_map=test_map)
        
        print(f"\n>>> Running Trajectory Analysis. Results will be saved to {results_path}...")
        
        # evaluator.evaluate automatically calls trainer.compute_action()
        evaluator.evaluate(trainer, results_path, episodes=25)