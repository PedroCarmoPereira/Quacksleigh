"""
Script for training lane-following agents (including collision avoidance) using Curriculum Learning.
Many properties of the training could be configured by modifying the config_updates dictionary at line 48.
For all available configuration options and their description, see config/config.yml and config/algo/ppo.yml
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"
###########################################################
# Imports
import os
import argparse
from datetime import datetime
import logging
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from ray.tune.logger import CSVLogger, TBXLogger

from config.paths import ArtifactPaths
from config.config import load_config, print_config, dump_config, update_config, find_and_load_config_by_seed

from duckietown_utils.env import launch_and_wrap_env
from duckietown_utils.utils import seed
from duckietown_utils.rllib_callbacks import on_episode_start, on_episode_step, on_episode_end
from duckietown_utils.rllib_callbacks import on_train_result as orig_on_train_result
from config.curriculum import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # BooleanOptionalAction lets you pass --curriculum or --no-curriculum
    parser.add_argument('--curriculum', dest='curriculum', action='store_true', default=True,
                    help='Use curriculum learning (default).')
    parser.add_argument('--no-curriculum', dest='curriculum', action='store_false',
                    help='Disable curriculum learning.')

    parser.add_argument('--resume-seed', default=-1, type=int, help='Seed to resume training from.')
    parser.add_argument('--resume-stage', default=0, type=int, help='Curriculum stage to resume from.')
    parser.add_argument('-e', '--experiment-idx', default=0, type=int,
                    help='Experiment Id')
    parser.add_argument('-c', '--checkpoint-idx', default=0, type=int,
                        help='Checkpoint Id')
    
    args = parser.parse_args()

    ###########################################################
    # Load config
    ###########################################################
    config = load_config('./config/config.yml', config_updates={"env_config": {"mode": "train"}})
    seed(1234)

    ###########################################################
    # Set up experiment parameters based on Curriculum Toggle
    ###########################################################

    resume_stage = 0
    if args.resume_stage:
        resume_stage = args.resume_stage

    if args.curriculum:
        # Start at Stage 0 parameters
        obs_density = CURRICULUM[resume_stage]['density']
        obs_static = CURRICULUM[resume_stage]['static']
        exp_name = "DomainRandomised_Curriculum"
        train_result_callback = custom_on_train_result
        logger.info(">>> Curriculum Learning ENABLED. Starting at Stage 0.")
    else:
        # Skip curriculum; go straight to target difficulty
        obs_density = 0.5
        obs_static = False
        exp_name = "DomainRandomised"
        train_result_callback = orig_on_train_result
        logger.info(">>> Curriculum Learning DISABLED. Training on full difficulty.")

    config_updates = {"seed": 1118,  
                    "experiment_name": exp_name,
                    "env_config": {"domain_rand": True,
                                    "dynamics_rand": True,
                                    "camera_rand": True,
                                    "grayscale_image": True,
                                    "spawn_obstacles": True,
                                    "resume_stage": resume_stage,
                                    "obstacles": {
                                        "duckie": {
                                            "density": obs_density,
                                            "static": obs_static,
                                        }
                                    },
                                    },
                    "rllib_config": {
                        "evaluation_interval": None,
                        "num_gpus": 0 
                    },
                    "timesteps_total": 2.e+6,
                    }
    update_config(config, config_updates)

    ###########################################################
    # Restore training
    ###########################################################
    restore_seed = args.resume_seed if args.resume_seed >= 0 else config.get('restore_seed', -1)
    if restore_seed >= 0:
        pretrained_config, checkpoint_path = \
            find_and_load_config_by_seed(restore_seed,
                                         experiment_name_filter="Curriculum" if args.curriculum else None,
                                         preselected_experiment_idx=config.get('restore_experiment_idx'),
                                         preselected_checkpoint_idx=config.get('restore_checkpoint_idx'))
        logger.warning("Overwriting config from {}".format(checkpoint_path))
        config = pretrained_config
        update_config(config, config_updates)
    else:
        checkpoint_path = None

    ###########################################################
    # Print config
    ###########################################################
    print_config(config)

    ###########################################################
    # Setup paths
    ###########################################################
    paths = ArtifactPaths(config['experiment_name'], config['seed'], algo_name=config['algo'])

    ###########################################################
    # Code backup
    ###########################################################
    os.system('cp -ar ./duckietown_utils {}/'.format(paths.code_backup_path))
    os.system('cp -ar ./experiments {}/'.format(paths.code_backup_path))
    os.system('cp -ar ./config {}/'.format(paths.code_backup_path))

    ###########################################################
    # Set up env and training config
    ###########################################################
    ray.init(**config["ray_init_config"])
    register_env('Duckietown', launch_and_wrap_env)

    config["rllib_config"].update({'env': 'Duckietown',
                                'callbacks': {'on_episode_start': on_episode_start,
                                              'on_episode_step': on_episode_step,
                                              'on_episode_end': on_episode_end,
                                              'on_train_result': train_result_callback},  # Injects the proper callback dynamically
                                "env_config": config["env_config"],
                                })
    dump_config(config, paths.experiment_base_path)

    ###########################################################
    # Run the training
    ###########################################################
    tune.run(PPOTrainer,
            stop={'timesteps_total': config["timesteps_total"]},
            config=config["rllib_config"],
            local_dir="./artifacts",
            checkpoint_at_end=True,
            trial_name_creator=lambda trial: trial.trainable_name,  
            name=paths.experiment_folder,
            keep_checkpoints_num=2,
            checkpoint_score_attr="episode_reward_mean",
            checkpoint_freq=1,
            restore=checkpoint_path,
            loggers=[CSVLogger, TBXLogger]
            )