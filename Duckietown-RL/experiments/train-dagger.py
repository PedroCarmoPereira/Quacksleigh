"""
Script to train a CNN policy using DAgger (Dataset Aggregation) by distilling 
the knowledge from a pre-trained PPO agent.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024"

import os
import logging
import numpy as np
import pandas as pd
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

os.environ['CUDA_VISIBLE_DEVICES'] = '' 
def build_novice_model(input_shape, action_dim):
    """
    Builds a smaller CNN for the Novice policy to distill knowledge 
    into a lightweight model for faster inference.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (8, 8), strides=(4, 4), padding='valid', 
                               activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (4, 4), strides=(2, 2), padding='valid', 
                               activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(action_dim, activation='linear')
    ])
    model.summary()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed-model-id', default=42, type=int,
                        help='Unique experiment identifier for the pre-trained PPO Expert.')
    parser.add_argument('--map-name', default='multimap1', help="Specify the training map")
    parser.add_argument('--iterations', default=10, type=int, help='Number of DAgger iterations')
    parser.add_argument('--episodes-per-iter', default=30, type=int, help='Episodes to roll out per iteration')
    parser.add_argument('--epochs', default=5, type=int, help='Epochs to train novice per iteration')
    parser.add_argument('--batch-size', default=64, type=int, help='Training batch size')
    args = parser.parse_args()

    seed(1234)

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

    expert_trainer = PPOTrainer(config=config["rllib_config"])
    expert_trainer.restore(checkpoint_path)

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

    rewards = {}
    cur_max_rwd = 0

    for it in range(args.iterations):
        print(f"\n=========================================")
        print(f"DAgger Iteration {it + 1}/{args.iterations} (Beta: {beta:.2f})")
        print(f"=========================================")
        rewards[it] = []
        # Rollout phase
        for ep in range(args.episodes_per_iter):
            obs = env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                expert_action = expert_trainer.compute_action(obs, explore=False)
                
                obs_batch = np.expand_dims(obs, axis=0)
                novice_action = novice_model.predict(obs_batch)[0]
                
                if np.random.rand() < beta:
                    action_to_take = expert_action
                else:
                    action_to_take = novice_action
            
                dataset_obs.append(obs)
                dataset_actions.append(expert_action)
                

                obs, reward, done, info = env.step(action_to_take)
                ep_reward += reward
            
            rewards[it].append(ep_reward)
            print(f"  [Iter {it+1}] Episode {ep+1}/{args.episodes_per_iter} completed. Reward: {ep_reward:.2f}")

        mean_rwd = np.mean(rewards[it])
        if mean_rwd > cur_max_rwd:
            cur_max_rwd = mean_rwd
            best_path = f"artifacts/dagger_model_seed_{SEED}_best.h5"
            novice_model.save(best_path)
            print(f"New best model saved (mean reward: {mean_rwd:.2f})")

        row = pd.DataFrame([{
            'iteration': it,
            'episode_reward_mean': mean_rwd,
            'episode_reward_max': np.max(rewards[it]),
            'episode_reward_min': np.min(rewards[it]),
            'beta': beta,
        }])
        csv_path = f'artifacts/dagger_model_seed_{SEED}_progress.csv'
        row.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
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
    save_path = f"artifacts/dagger_model_seed_{SEED}.h5"
    novice_model.save(save_path)
    print(f"\n>>> DAgger training complete! Model saved to: {save_path}")
    print(len(dataset_actions))

    