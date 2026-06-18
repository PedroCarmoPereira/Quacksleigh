"""
Script to extract the base TensorFlow/Keras model from an RLlib PPO checkpoint 
and save it to an H5 file.
"""
import os
import argparse
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
import tensorflow as tf

from config.config import find_and_load_config_by_seed, update_config
from config.curriculum import *

from duckietown_utils.rllib_callbacks import *
from duckietown_utils.env import launch_and_wrap_env

def main():
    parser = argparse.ArgumentParser(description="Export RLlib checkpoint to H5 format.")
    parser.add_argument('-s', '--seed-model-id', required=True, type=int, 
                        help='Unique experiment identifier (seed) of the trained model')
    parser.add_argument('-o', '--output', default='exported_model.h5', type=str, 
                        help='Output .h5 file path')
    parser.add_argument('--curriculum', dest='curriculum', action='store_true', default=True,
                    help='Use curriculum learning (default).')
    parser.add_argument('--no-curriculum', dest='curriculum', action='store_false',
                    help='Disable curriculum learning.')
    args = parser.parse_args()

    print(f">>> Locating checkpoint for seed {args.seed_model_id}...")
    
    filt = None
    if args.curriculum:
        filt = 'Curriculum'
    # 1. Find and load the configuration and checkpoint path
    config, checkpoint_path = find_and_load_config_by_seed(args.seed_model_id, preselected_experiment_idx=0, preselected_checkpoint_idx=0, experiment_name_filter=filt)
    
    # Optional: Force inference mode so we don't allocate unnecessary training resources
    update_config(config, {'env_config': {'mode': 'inference'}})

    # 2. Initialize Ray and register the environment
    ray.init(**config.get("ray_init_config", {}))
    register_env('Duckietown', launch_and_wrap_env)

    # 3. Instantiate the PPOTrainer and restore the checkpoint
    print(f">>> Restoring checkpoint from: {checkpoint_path}")
    trainer = PPOTrainer(config=config["rllib_config"])
    trainer.restore(checkpoint_path)

    # 4. Extract the base Keras model from the RLlib policy
    print(">>> Extracting base TensorFlow model...")
    policy = trainer.get_policy()
    base_model = policy.model.base_model
    
    # Print a quick summary to verify the architecture
    base_model.summary()

    # 5. Save the model to H5
    base_model.save(args.output)
    print(f"\n>>> Successfully saved base model to {args.output}")

    # Shutdown ray
    ray.shutdown()

if __name__ == "__main__":
    main()