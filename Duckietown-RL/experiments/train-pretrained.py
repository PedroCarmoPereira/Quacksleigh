"""
Script for training lane-following agents using OpenCV's DNN module 
to extract features from a pre-trained YOLO ONNX backbone.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024"

import os
import argparse
import logging
import numpy as np
import cv2
import gym
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.tune.logger import CSVLogger, TBXLogger

from config.paths import ArtifactPaths
from config.config import load_config, print_config, dump_config, update_config, find_and_load_config_by_seed
from config.curriculum import *

from duckietown_utils.env import launch_and_wrap_yolo_env
from duckietown_utils.utils import seed
from duckietown_utils.rllib_callbacks import on_episode_start, on_episode_step, on_episode_end, on_train_result
from duckietown_utils.rllib_loggers import TensorboardImageLogger, WeightsAndBiasesLogger

logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = '' 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-onnx-path', type=str, default='artifacts/model_yolo/best.onnx', 
                        help='Path to the ONNX YOLO model')
    parser.add_argument('--curriculum', dest='curriculum', action='store_true', default=True,
                    help='Use curriculum learning (default).')
    parser.add_argument('--no-curriculum', dest='curriculum', action='store_false',
                    help='Disable curriculum learning.')
    parser.add_argument('--resume-seed', default=-1, type=int, help='Seed to resume training from.')
    parser.add_argument('--resume-stage', default=0, type=int, help='Curriculum stage to resume from.')
    args = parser.parse_args()
    args.yolo_onnx_path = os.path.abspath(args.yolo_onnx_path)

    ###########################################################
    # Load config
    config = load_config('./config/config.yml', config_updates={"env_config": {"mode": "train"}})
    seed(1234)

    resume_stage = args.resume_stage

    if args.curriculum:
        # Start at Stage 0 parameters
        obs_density = CURRICULUM[resume_stage]['density']
        obs_static = CURRICULUM[resume_stage]['static']
        exp_name = "Yolo_CV2_Backbone_PPO_Curriculum"
        train_result_callback = custom_on_train_result
        logger.info(">>> Curriculum Learning ENABLED. Starting at Stage 0.")
    else:
        # Skip curriculum; go straight to target difficulty
        obs_density = 0.5
        obs_static = False
        exp_name = "Yolo_CV2_Backbone_PPO"
        train_result_callback = on_train_result
        logger.info(">>> Curriculum Learning DISABLED. Training on full difficulty.")

    ###########################################################
    # Set up experiment parameters
    config_updates = {
        "seed": 1118,
        "experiment_name": exp_name,
        "timesteps_total": 1500000,
        "env_config": {
            "yolo_onnx_path": args.yolo_onnx_path,
            # Pass full RGB images to the wrapper so YOLO can process them properly
            "grayscale_image": False,
            "domain_rand": True,
            "dynamics_rand": True,
            "camera_rand": True,
            "spawn_obstacles": True,
            "resume_stage": resume_stage,
            "obstacles": {
                "duckie": {
                    "density": obs_density,
                    "static": obs_static,
                }
            }
        },
        "rllib_config": {
            # RLlib will automatically detect that the observation space is now a 1D vector 
            # (thanks to our wrapper) and will automatically build a fast Fully-Connected 
            # network instead of a CNN! We don't even need a custom model class.
            "num_gpus": 0,
            "evaluation_interval": None,
            "model": {
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu"
            }
        }
    }
    update_config(config, config_updates)

    restore_seed = args.resume_seed if args.resume_seed >= 0 else config.get('restore_seed', -1)
    if restore_seed >= 0:
        pretrained_config, checkpoint_path = \
            find_and_load_config_by_seed(restore_seed,
                                         experiment_name_filter=exp_name,
                                         preselected_experiment_idx=0,
                                         preselected_checkpoint_idx=1)
        logger.warning("Overwriting config from {}".format(checkpoint_path))
        config = pretrained_config
        update_config(config, config_updates)
    else:
        checkpoint_path = None

    ###########################################################
    print_config(config)

    ###########################################################
    # Setup paths
    paths = ArtifactPaths(config['experiment_name'], config['seed'], algo_name=config['algo'])

    # Code backup
    os.system(f'cp -ar ./duckietown_utils {paths.code_backup_path}/')
    os.system(f'cp -ar ./experiments {paths.code_backup_path}/')
    os.system(f'cp -ar ./config {paths.code_backup_path}/')

    ###########################################################
    # Set up env and training config
    ray.init(**config["ray_init_config"])
    
    # Register our modified environment launcher
    register_env('Duckietown', launch_and_wrap_yolo_env)
    
    config["rllib_config"].update({
        'env': 'Duckietown',
        'callbacks': {
            'on_episode_start': on_episode_start,
            'on_episode_step': on_episode_step,
            'on_episode_end': on_episode_end,
            'on_train_result': train_result_callback
        },
        "env_config": config["env_config"],
    })
    
    dump_config(config, paths.experiment_base_path)

    ###########################################################
    # Run the training
    tune.run(
        PPOTrainer,
        stop={'timesteps_total': config["timesteps_total"]},
        config=config["rllib_config"],
        local_dir="./artifacts",
        checkpoint_at_end=True,
        trial_name_creator=lambda trial: trial.trainable_name,
        name=paths.experiment_folder,
        keep_checkpoints_num=1,
        checkpoint_score_attr="episode_reward_mean",
        checkpoint_freq=1,
        loggers=[CSVLogger, TBXLogger]
    )