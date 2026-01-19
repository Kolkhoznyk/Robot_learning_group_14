import numpy as np
import gymnasium as gym

class SingleTaskOneHotWrapper(gym.Env):
    """
    Wrapper to append task id to observation (onehot-encoding)
    """
    def __init__(self, env, task_id, n_tasks, reward_scales):
        super().__init__()
        self.env = env
        self.task_id = task_id
        self.n_tasks = n_tasks
        self.reward_scales = reward_scales

        obs_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([obs_space.low, np.zeros(self.n_tasks)]),
            high=np.concatenate([obs_space.high, np.ones(self.n_tasks)]),
            dtype=np.float32
        )
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._one_hot_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.reward_scales is not None:
            reward = reward * self.reward_scales[self.task_id]
        return self._one_hot_obs(obs), reward, terminated, truncated, info

    def _one_hot_obs(self, obs):
        one_hot = np.zeros(self.n_tasks, dtype=np.float32)
        one_hot[self.task_id] = 1.0
        return np.concatenate([np.array(obs, dtype=np.float32), one_hot])