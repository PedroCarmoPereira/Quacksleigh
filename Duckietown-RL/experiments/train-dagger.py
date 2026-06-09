"""
Script to train a CNN policy using DAgger (Dataset Aggregation) by distilling 
the knowledge from a pre-trained PPO agent.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024"

import os
import logging
import numpy as np
import argparse
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
import tensorflow as tf

from config.config import find_and_load_config_by_seed, update_config, print_config
from duckietown_utils.env import launch_and_wrap_env
from duckietown_utils.utils import seed
from duckietown_utils.rllib_callbacks import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Set to a GPU ID if you want to use GPU for training

def build_novice_model(input_shape, action_dim):
    """
    Builds a smaller CNN for the Novice policy to distill knowledge 
    into a lightweight model for faster inference.
    """
    model = tf.keras.Sequential([
        # 1st Conv layer (Matches expert: 16 filters, 8x8 kernel, stride 4)
        # Input: (84, 84, 3) -> Output: (20, 20, 16)
        tf.keras.layers.Conv2D(16, (8, 8), strides=(4, 4), padding='valid', 
                               activation='relu', input_shape=input_shape),
        
        # 2nd Conv layer (Matches expert: 32 filters, 4x4 kernel, stride 2)
        # Input: (20, 20, 16) -> Output: (9, 9, 32)
        tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='valid', 
                               activation='relu'),
        
        # Flatten to 1D (Size: 9 * 9 * 32 = 2592)
        tf.keras.layers.Flatten(),
        
        # Smaller Dense layer instead of the Expert's massive 256-channel Conv2D bottleneck
        tf.keras.layers.Dense(64, activation='relu'),
        
        # Output action layer
        tf.keras.layers.Dense(action_dim, activation='linear')
    ])
    
    # Print the summary so you can verify the parameter reduction!
    model.summary()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    return model
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed-model-id', default=42, type=int,
                        help='Unique experiment identifier for the pre-trained PPO Expert.')
    parser.add_argument('--map-name', default='multimap1', help="Specify the training map")
    parser.add_argument('--iterations', default=10, type=int, help='Number of DAgger iterations')
    parser.add_argument('--episodes-per-iter', default=15, type=int, help='Episodes to roll out per iteration')
    parser.add_argument('--epochs', default=5, type=int, help='Epochs to train novice per iteration')
    parser.add_argument('--batch-size', default=64, type=int, help='Training batch size')
    args = parser.parse_args()

    # Set numpy and random seed
    seed(1234)

    # 1. Load Expert (PPO) Config & Environment
    print(">>> Loading pre-trained PPO Expert...")
    SEED = args.seed_model_id
    config, checkpoint_path = find_and_load_config_by_seed(SEED, preselected_experiment_idx=2, preselected_checkpoint_idx=0)
    
    update_config(config, {
        'env_config': {
            'mode': 'inference',
            'training_map': args.map_name,
            "domain_rand": True,
            "dynamics_rand": True,
            "camera_rand": True,
            "grayscale_image":True,
            "spawn_obstacles": True,
            "obstacles": {
                "duckie": {
                    "density": 0.5,
                    "static": False,
                }
            }
        }
    })

    ray.init(**config["ray_init_config"])
    register_env('Duckietown', launch_and_wrap_env)

    # 2. Restore Expert Agent
    expert_trainer = PPOTrainer(config=config["rllib_config"])
    expert_trainer.restore(checkpoint_path)

    # 3. Setup Environment & Novice Model
    env = launch_and_wrap_env(config["env_config"])
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    
    print(f">>> Initializing Novice CNN (Input: {obs_shape}, Output: {action_dim})...")
    novice_model = build_novice_model(obs_shape, action_dim)

    # 4. DAgger Loop
    dataset_obs = []
    dataset_actions = []
    beta = 1.0  # Probability of using the expert's action during rollout
    beta_decay = 0.8 # Decay factor to slowly transition control to the novice

    for it in range(args.iterations):
        print(f"\n=========================================")
        print(f"DAgger Iteration {it + 1}/{args.iterations} (Beta: {beta:.2f})")
        print(f"=========================================")
        
        # Rollout phase
        for ep in range(args.episodes_per_iter):
            obs = env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                # 4.a: Get Expert Action (Always recorded for ground truth)
                expert_action = expert_trainer.compute_action(obs, explore=False)
                
                # 4.b: Get Novice Action
                obs_batch = np.expand_dims(obs, axis=0)
                novice_action = novice_model.predict(obs_batch)[0]
                
                # 4.c: Mix Policies for environmental stepping (DAgger specific)
                if np.random.rand() < beta:
                    action_to_take = expert_action
                else:
                    action_to_take = novice_action
                
                # 4.d: Aggregate dataset (Always map observation to EXPERT action)
                dataset_obs.append(obs)
                dataset_actions.append(expert_action)
                
                # Step environment
                obs, reward, done, info = env.step(action_to_take)
                ep_reward += reward
                
            print(f"  [Iter {it+1}] Episode {ep+1}/{args.episodes_per_iter} completed. Reward: {ep_reward:.2f}")

        # Training phase
        print(f">>> Training Novice CNN on aggregated dataset (Total samples: {len(dataset_obs)})...")
        X_train = np.array(dataset_obs)
        y_train = np.array(dataset_actions)
        
        novice_model.fit(
            X_train, y_train, 
            batch_size=args.batch_size, 
            epochs=args.epochs, 
            validation_split=0.1,
            shuffle=True
        )

        # Decay the probability of using the expert policy
        beta *= beta_decay

    # 5. Save the trained Novice Model
    os.makedirs("artifacts", exist_ok=True)
    save_path = f"artifacts/dagger_novice_model_seed{SEED}.h5"
    novice_model.save(save_path)
    print(f"\n>>> DAgger training complete! Model saved to: {save_path}")