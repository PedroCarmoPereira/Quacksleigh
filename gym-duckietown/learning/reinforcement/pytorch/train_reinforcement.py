import argparse
import logging

import os
import numpy as np

# Duckietown Specific
from reinforcement.pytorch.ddpg import DDPG
from reinforcement.pytorch.rl_utils import seed, evaluate_policy, ReplayBuffer, run_policy_demo, plot_training_curves
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _train(args):
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

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

    # Set seeds
    seed(args.seed)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = DDPG(
        state_dim,
        action_dim,
        max_action,
        net_type="cnn",
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
    )
    replay_buffer = ReplayBuffer(args.replay_buffer_max_size)
    print("Initialized DDPG")

    # Evaluate untrained policy
    evaluations = [evaluate_policy(env, policy)]
    eval_timesteps = [0]
    best_eval = evaluations[0]
    early_stop_counter = 0
    collapse_counter = 0

    # Metric accumulators for plotting
    ep_rewards_log = []
    ep_timesteps_log = []
    ep_lengths_log = []
    critic_losses_log = []
    actor_losses_log = []

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    episode_reward = None
    env_counter = 0
    reward = 0
    episode_timesteps = 0
    print("Starting training")
    while total_timesteps < args.max_timesteps:

        if args.log_freq > 0 and total_timesteps % args.log_freq == 0:
            print("timestep: {} | reward: {}".format(total_timesteps, reward))

        if done:
            if total_timesteps != 0:
                print(
                    ("Total T: %d Episode Num: %d Episode T: %d Reward: %f")
                    % (total_timesteps, episode_num, episode_timesteps, episode_reward)
                )
                ep_rewards_log.append(episode_reward)
                ep_timesteps_log.append(total_timesteps)
                ep_lengths_log.append(episode_timesteps)

                buffer_size = len(replay_buffer.storage)
                if buffer_size >= args.batch_size:
                    train_iters = min(args.train_iters, buffer_size)
                    mean_critic_loss, mean_actor_loss = policy.train(
                        replay_buffer, train_iters, args.batch_size, args.discount, args.tau
                    )
                    critic_losses_log.append(mean_critic_loss)
                    actor_losses_log.append(mean_actor_loss)

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    eval_reward = evaluate_policy(env, policy)
                    evaluations.append(eval_reward)
                    eval_timesteps.append(total_timesteps)
                    print("rewards at time {}: {}".format(total_timesteps, eval_reward))

                    if args.save_models:
                        policy.save(filename="ddpg", directory=args.model_dir)
                    np.savez("./results/rewards.npz", evaluations)

                    # Track improvement and save best checkpoint separately
                    improvement = eval_reward - best_eval
                    if improvement > 0:
                        prev_best = best_eval
                        best_eval = eval_reward
                        if args.save_models:
                            policy.save(filename="ddpg_best", directory=args.model_dir)
                            print(
                                "New best eval reward: {:.4f} (prev best: {:.4f}). "
                                "Saved ddpg_best_*.pth.".format(eval_reward, prev_best)
                            )

                    # Early-stopping bookkeeping (independent of best-saving)
                    if improvement >= args.early_stop_min_delta:
                        early_stop_counter = 0
                    else:
                        early_stop_counter += 1

                    # Collapse detector: did eval reward drop far below best and stay there?
                    if best_eval > 0 and eval_reward < args.collapse_threshold * best_eval:
                        collapse_counter += 1
                    else:
                        collapse_counter = 0

                    plot_training_curves(
                        eval_timesteps=eval_timesteps,
                        eval_rewards=evaluations,
                        ep_timesteps=ep_timesteps_log,
                        ep_rewards=ep_rewards_log,
                        ep_lengths=ep_lengths_log,
                        critic_losses=critic_losses_log,
                        actor_losses=actor_losses_log,
                        env_timesteps=args.env_timesteps,
                    )

                    if args.reward_threshold is not None and eval_reward >= args.reward_threshold:
                        print(
                            "Reward threshold reached: {:.4f} >= {:.4f}. Stopping.".format(
                                eval_reward, args.reward_threshold
                            )
                        )
                        break

                    if collapse_counter >= args.collapse_patience:
                        print(
                            "Collapse detected: {} consecutive evals below {:.0%} of best "
                            "({:.4f}). Stopping.".format(
                                args.collapse_patience,
                                args.collapse_threshold,
                                best_eval,
                            )
                        )
                        break

                    if (
                        args.early_stop
                        and total_timesteps >= args.early_stop_min_timesteps
                        and early_stop_counter >= args.early_stop_patience
                    ):
                        print(
                            "Early stopping: no eval improvement of {:.4f} "
                            "for {} consecutive evals (best: {:.4f})".format(
                                args.early_stop_min_delta,
                                args.early_stop_patience,
                                best_eval,
                            )
                        )
                        break

            # Reset environment
            env_counter += 1
            obs = reset_env(env)
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.predict(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high
                )

        # Perform action
        new_obs, reward, done, _ = step_env(env, action)

        if episode_timesteps >= args.env_timesteps:
            done = True

        done_bool = 0 if episode_timesteps + 1 == args.env_timesteps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add(obs, new_obs, action, reward, done_bool)

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    print("Training done, about to save..")
    policy.save(filename="ddpg", directory=args.model_dir)
    print("Finished saving.")

    plot_training_curves(
        eval_timesteps=eval_timesteps,
        eval_rewards=evaluations,
        ep_timesteps=ep_timesteps_log,
        ep_rewards=ep_rewards_log,
        ep_lengths=ep_lengths_log,
        critic_losses=critic_losses_log,
        actor_losses=actor_losses_log,
        env_timesteps=args.env_timesteps,
    )

    if args.demo_after_train:
        run_policy_demo(env, policy, max_steps=args.demo_steps, expl_noise=0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # DDPG Args
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    # currently running at 5k but I think this may be too much
    parser.add_argument(
        "--start_timesteps", default=2500, type=int
    )  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument(
        "--max_timesteps", default=50000, type=int
    )  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true", default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=512, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument(
        "--policy_noise", default=0.2, type=float
    )  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--env_timesteps", default=500, type=int)  # Max steps per episode
    parser.add_argument(
        "--train_iters", default=200, type=int
    )  # Gradient steps per episode end
    parser.add_argument(
        "--log_freq", default=200, type=int
    )  # Print every N env steps (0 to disable per-step logging)
    parser.add_argument(
        "--replay_buffer_max_size", default=10000, type=int
    )  # Maximum number of steps to keep in the replay buffer
    parser.add_argument("--early_stop", action="store_true", default=True)
    parser.add_argument(
        "--early_stop_patience", default=5, type=int
    )  # Consecutive evals without improvement before stopping
    parser.add_argument(
        "--early_stop_min_delta", default=1.0, type=float
    )  # Minimum eval reward increase to count as improvement
    parser.add_argument(
        "--early_stop_min_timesteps", default=2.5e4, type=int
    )  # Early stopping cannot fire before this many total timesteps
    parser.add_argument(
        "--reward_threshold", default=None, type=float
    )  # Stop as soon as eval reward reaches this value (None = disabled)
    parser.add_argument(
        "--collapse_threshold", default=0.5, type=float
    )  # Fraction of best_eval below which a policy is considered collapsed
    parser.add_argument(
        "--collapse_patience", default=5, type=int
    )  # Consecutive collapsed evals before stopping
    parser.add_argument(
        "--actor_lr", default=1e-4, type=float
    )  # Adam learning rate for the actor network
    parser.add_argument(
        "--critic_lr", default=1e-3, type=float
    )  # Adam learning rate for the critic network
    parser.add_argument("--model-dir", type=str, default="reinforcement/pytorch/models/")
    parser.add_argument(
        "--demo_after_train",
        action="store_true",
        default=True,
        help="Run a headful GUI rollout after training (default: on)",
    )
    parser.add_argument(
        "--no_demo_after_train",
        dest="demo_after_train",
        action="store_false",
        help="Skip the post-training GUI demo",
    )
    parser.add_argument(
        "--demo_steps",
        default=500,
        type=int,
        help="Number of env steps for the post-training GUI demo",
    )

    _train(parser.parse_args())
