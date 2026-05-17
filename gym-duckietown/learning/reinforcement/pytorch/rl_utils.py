import os
import random

import gym
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless — works without a display
import matplotlib.pyplot as plt

from utils.wrappers import reset_env, step_env


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, reward, done):
        data = (state, next_state, action, reward, done)
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        else:
            # FIFO eviction via circular pointer (O(1), no list shift)
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size=100, flat=True):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]

            if flat:
                states.append(np.array(state, copy=False).flatten())
                next_states.append(np.array(next_state, copy=False).flatten())
            else:
                states.append(np.array(state, copy=False))
                next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        # state_sample, action_sample, next_state_sample, reward_sample, done_sample
        return {
            "state": np.stack(states),
            "next_state": np.stack(next_states),
            "action": np.stack(actions),
            "reward": np.stack(rewards).reshape(-1, 1),
            "done": np.stack(dones).reshape(-1, 1),
        }


def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        obs = reset_env(env)
        done = False
        step = 0
        while not done and step < max_timesteps:
            action = policy.predict(np.array(obs))
            obs, reward, done, _ = step_env(env, action)
            avg_reward += reward
            step += 1

    avg_reward /= eval_episodes

    return avg_reward


def plot_training_curves(
    eval_timesteps,
    eval_rewards,
    ep_timesteps,
    ep_rewards,
    critic_losses,
    actor_losses,
    ep_lengths=None,
    env_timesteps=None,
    save_path="./results/training_curves.png",
):
    """Save a 2x3 figure tracking eval reward, episode reward, episode length, critic loss, actor loss."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("DDPG Training Curves", fontsize=14)

    # [0,0]: evaluation reward vs total timesteps
    ax = axes[0, 0]
    if eval_timesteps and eval_rewards:
        ax.plot(eval_timesteps, eval_rewards, marker="o", linewidth=1.5)
    ax.set_title("Eval Reward (avg over 10 episodes)")
    ax.set_xlabel("Total timesteps")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    # [0,1]: per-episode training reward vs episode index
    ax = axes[0, 1]
    if ep_rewards:
        ax.plot(ep_rewards, linewidth=1.0, alpha=0.7, label="episode")
        # smooth with a rolling window if there are enough episodes
        window = max(1, len(ep_rewards) // 10)
        if len(ep_rewards) >= window:
            smoothed = np.convolve(ep_rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(ep_rewards)), smoothed, linewidth=2.0, label=f"smoothed (w={window})")
        ax.legend(fontsize=8)
    ax.set_title("Training Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)

    # [0,2]: episode length vs episode index — shows whether episodes hit the env_timesteps cap
    ax = axes[0, 2]
    if ep_lengths:
        ax.plot(ep_lengths, linewidth=1.0, alpha=0.7, color="green", label="episode length")
        window = max(1, len(ep_lengths) // 10)
        if len(ep_lengths) >= window:
            smoothed = np.convolve(ep_lengths, np.ones(window) / window, mode="valid")
            ax.plot(
                range(window - 1, len(ep_lengths)),
                smoothed,
                linewidth=2.0,
                color="darkgreen",
                label=f"smoothed (w={window})",
            )
        if env_timesteps is not None:
            ax.axhline(
                y=env_timesteps,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"cap ({env_timesteps})",
            )
        ax.legend(fontsize=8)
    ax.set_title("Episode Length (steps until done)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.grid(True, alpha=0.3)

    # [1,0]: critic loss vs training-call index
    ax = axes[1, 0]
    if critic_losses:
        ax.plot(critic_losses, linewidth=1.0, alpha=0.8)
    ax.set_title("Critic Loss (mean per train call)")
    ax.set_xlabel("Train call index")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)

    # [1,1]: actor loss vs training-call index
    ax = axes[1, 1]
    if actor_losses:
        ax.plot(actor_losses, linewidth=1.0, alpha=0.8, color="orange")
    ax.set_title("Actor Loss (mean per train call)")
    ax.set_xlabel("Train call index")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # [1,2]: hide unused cell
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
    print("Saved training curves to {}".format(save_path))


def run_policy_demo(env, policy, max_steps=500, expl_noise=0.0):
    """Run the policy with env.render() for a headful GUI preview."""
    obs = reset_env(env)
    done = False
    total_reward = 0.0

    print("Starting GUI demo for {} steps (close the window or Ctrl+C to stop early).".format(max_steps))

    for step in range(max_steps):
        action = policy.predict(np.array(obs))
        if expl_noise:
            action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(
                env.action_space.low, env.action_space.high
            )

        obs, reward, done, _ = step_env(env, action)
        total_reward += reward
        env.render()

        if (step + 1) % 50 == 0:
            print("Demo step {}/{} | cumulative reward: {:.2f}".format(step + 1, max_steps, total_reward))

        if done:
            obs = reset_env(env)
            done = False

    print("Demo finished: {} steps, cumulative reward: {:.2f}".format(max_steps, total_reward))
