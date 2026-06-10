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
from config.curriculum import *
from duckietown_utils.env import launch_and_wrap_env
from duckietown_utils.utils import seed
from duckietown_utils.rllib_callbacks import *

from collections import deque

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
    # BooleanOptionalAction lets you pass --curriculum or --no-curriculum
    parser.add_argument('--curriculum', dest='curriculum', action='store_true', default=True,
                    help='Use curriculum learning (default).')
    parser.add_argument('--no-curriculum', dest='curriculum', action='store_false',
                    help='Disable curriculum learning.')
    parser.add_argument('-s', '--seed-model-id', default=42, type=int)
    parser.add_argument('--map-name', default='multimap1')
    parser.add_argument('--iterations', default=30, type=int)
    parser.add_argument('--episodes-per-iter', default=30, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
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
            "grayscale_image": True,
            "spawn_obstacles": True,
            "obstacles": {"duckie": {"density": 0.5, "static": False}}
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

    os.makedirs("artifacts", exist_ok=True)


    MAX_BUFFER_SIZE = 50000 
    dataset_obs     = deque(maxlen=MAX_BUFFER_SIZE)
    dataset_actions = deque(maxlen=MAX_BUFFER_SIZE)
    beta            = 1.0
    beta_decay      = 0.8
    BETA_BUMP       = 0.3
    rewards         = {}
    cur_max_rwd     = 0

    stage_idx      = 0
    iters_at_stage = 0

    if args.curriculum:
        advance_stage(config["env_config"], CURRICULUM[stage_idx])
        print(f">>> Starting curriculum stage 0: {CURRICULUM[0]['label']}")

    it = 0
    while it < args.iterations:
        stage = CURRICULUM[stage_idx] if args.curriculum else None

        print(f"\n{'='*45}")
        if args.curriculum:
            print(f"DAgger Iter {it+1}/{args.iterations} | Stage {stage_idx}: {stage['label']} | Beta: {beta:.2f}")
        else:
            print(f"DAgger Iter {it+1}/{args.iterations} | Beta: {beta:.2f}")
        print(f"{'='*45}")

        # Rollout into TEMP buffers — we decide whether to keep this data after the stage check
        iter_obs     = []
        iter_actions = []
        iter_rewards = []

        for ep in range(args.episodes_per_iter):
            obs   = env.reset()
            done  = False
            ep_reward = 0

            while not done:
                expert_action = expert_trainer.compute_action(obs, explore=False)
                obs_batch     = np.expand_dims(obs, axis=0)
                novice_action = novice_model.predict(obs_batch)[0]

                action_to_take = expert_action if np.random.rand() < beta else novice_action

                iter_obs.append(obs)
                iter_actions.append(expert_action)

                obs, reward, done, info = env.step(action_to_take)
                ep_reward += reward

            iter_rewards.append(ep_reward)
            print(f"  [Iter {it+1}] Episode {ep+1}/{args.episodes_per_iter} | Reward: {ep_reward:.2f}")

        mean_rwd = np.mean(iter_rewards)

        # --- Curriculum advancement check (before training) ---
        stage_changed = False
        if args.curriculum:
            next_stage_idx = stage_idx + 1
            threshold      = stage['min_reward']
            can_advance    = (
                threshold is not None
                and next_stage_idx < len(CURRICULUM)
                and mean_rwd >= threshold
            )

            if can_advance:
                iters_at_stage += 1
                print(f"  [Curriculum] Above threshold ({mean_rwd:.1f} >= {threshold}) "
                      f"for {iters_at_stage}/{STAGE_STABILITY} iters.")

                if iters_at_stage >= STAGE_STABILITY:
                    stage_idx      += 1
                    iters_at_stage  = 0
                    beta            = min(1.0, beta + BETA_BUMP)
                    stage_changed   = True

                    advance_stage(config["env_config"], CURRICULUM[stage_idx])

                    # Discard all old-stage data — fresh start on new difficulty
                    dataset_obs.clear()
                    dataset_actions.clear()

                    print(f"  >>> ADVANCING to stage {stage_idx}: {CURRICULUM[stage_idx]['label']}")
                    print(f"  >>> Dataset cleared. Beta bumped to {beta:.2f}. Restarting iteration.")
            else:
                iters_at_stage = 0

        if stage_changed:
            # Don't train, don't log, don't increment it, don't decay beta
            continue

        # Stage didn't change — commit this iteration's data to the main dataset
        dataset_obs.extend(iter_obs)
        dataset_actions.extend(iter_actions)
        rewards[it] = iter_rewards

        # Checkpointing
        if mean_rwd > cur_max_rwd:
            cur_max_rwd = mean_rwd
            if args.curriculum:
                novice_model.save(f"artifacts/dagger_model_seed_{SEED}_cur_best.h5")
            else:
                novice_model.save(f"artifacts/dagger_model_seed_{SEED}_best.h5")
            print(f"New best model saved (mean reward: {mean_rwd:.2f})")

        # Incremental CSV
        row = pd.DataFrame([{
            'iteration':           it,
            'stage':               stage_idx if args.curriculum else 0,
            'episode_reward_mean': mean_rwd,
            'episode_reward_max':  np.max(iter_rewards),
            'episode_reward_min':  np.min(iter_rewards),
            'beta':                beta,
        }])
        csv_path = f'artifacts/dagger_model_seed_{SEED}_progress.csv'
        row.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

        # Training phase
        print(f">>> Training Novice on aggregated dataset ({len(dataset_obs)} samples)...")
        X_train = np.array(dataset_obs)
        y_train = np.array(dataset_actions)

        novice_model.fit(
            X_train, y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=0.1,
            shuffle=True
        )

        beta = beta * beta_decay
        it += 1

    # Final save
    if args.curriculum:
        save_path = f"artifacts/dagger_model_seed_{SEED}_cur.h5"
    else:
        save_path = f"artifacts/dagger_model_seed_{SEED}.h5"
    novice_model.save(save_path)
    print(f"\n>>> DAgger training complete! Model saved to: {save_path}")

    env.close()