import gym
from gym import spaces
import numpy as np

from gym_duckietown.simulator import Simulator


def reset_env(env, **kwargs):
    """Return observation only; works with gym<0.26 and gym>=0.26."""
    result = env.reset(**kwargs)
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    """Return (obs, reward, done, info); works with gym<0.26 and gym>=0.26."""
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info
    return result


class LegacyGymCompatWrapper(gym.Wrapper):
    """Adapt duckietown's legacy env API for gym>=0.26 wrappers."""

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            if info is None:
                info = {}
            return obs, reward, bool(done), False, info
        return result


class MotionBlurWrapper(Simulator):
    def __init__(self, env=None):
        Simulator.__init__(self)
        self.env = env
        self.frame_skip = 3
        self.env.delta_time = self.env.delta_time / self.frame_skip

    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        # Actions could be a Python list
        action = np.array(action)
        motion_blur_window = []
        for _ in range(self.frame_skip):
            obs = self.env.render_obs()
            motion_blur_window.append(obs)
            self.env.update_physics(action)

        # Generate the current camera image

        obs = self.env.render_obs()
        motion_blur_window.append(obs)
        obs = np.average(motion_blur_window, axis=0, weights=[0.8, 0.15, 0.04, 0.01])

        misc = self.env.get_agent_info()

        d = self.env._compute_done_reward()
        misc["Simulator"]["msg"] = d.done_why

        return obs, d.reward, d.done, misc


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        obs_space = self.observation_space
        self.observation_space = spaces.Box(
            obs_space.low[0, 0, 0],
            obs_space.high[0, 0, 0],
            shape,
            dtype=obs_space.dtype,
        )
        self.shape = shape

    def observation(self, observation):
        from PIL import Image

        height, width = self.shape[0], self.shape[1]
        resized = Image.fromarray(observation).resize((width, height), Image.BILINEAR)
        return np.asarray(resized)


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
        else:
            reward += 4

        return reward


# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_
