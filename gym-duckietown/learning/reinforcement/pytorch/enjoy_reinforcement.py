import argparse

import numpy as np

# Duckietown Specific
from reinforcement.pytorch.ddpg import DDPG
from utils.env import launch_env
from utils.wrappers import (
    NormalizeWrapper,
    ImgWrapper,
    DtRewardWrapper,
    ActionWrapper,
    ResizeWrapper,
    reset_env,
    step_env,
)


def _enjoy(args):
    # Launch the env with our helper function
    env = launch_env()
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy and load weights
    policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    print("Loading model '{}' from '{}'".format(args.filename, args.directory))
    policy.load(filename=args.filename, directory=args.directory)

    episode_rewards = []
    episode_steps = []

    for ep in range(args.episodes):
        obs = reset_env(env)
        done = False
        ep_reward = 0.0
        step = 0
        while not done and step < args.max_steps:
            action = policy.predict(np.array(obs))
            obs, reward, done, _ = step_env(env, action)
            ep_reward += reward
            if args.render:
                env.render()
            step += 1

        episode_rewards.append(ep_reward)
        episode_steps.append(step)
        print(
            "Episode {}/{}: steps={}, cumulative reward={:.2f}".format(
                ep + 1, args.episodes, step, ep_reward
            )
        )

    rewards = np.array(episode_rewards, dtype=np.float64)
    print(
        "\nResults over {} episodes  ->  mean: {:.2f}  std: {:.2f}  min: {:.2f}  max: {:.2f}".format(
            args.episodes,
            float(rewards.mean()),
            float(rewards.std()),
            float(rewards.min()),
            float(rewards.max()),
        )
    )
    print(
        "Average episode length: {:.1f} steps (max allowed: {})".format(
            float(np.mean(episode_steps)), args.max_steps
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        default="ddpg",
        help="Model filename prefix to load (e.g. 'ddpg' for latest or 'ddpg_best' for best).",
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="reinforcement/pytorch/models/",
        help="Directory containing the model files.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum steps per episode (matches training env_timesteps default).",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Render the GUI while evaluating (default: on).",
    )
    parser.add_argument(
        "--no_render",
        dest="render",
        action="store_false",
        help="Disable GUI rendering (faster, useful for headless verification).",
    )

    _enjoy(parser.parse_args())
