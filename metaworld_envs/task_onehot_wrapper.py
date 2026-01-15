# task_onehot_wrapper.py
import numpy as np
import gymnasium as gym


class TaskOneHotObsWrapper(gym.ObservationWrapper):
    """
    Appends a one-hot encoding of the current task_id to the observation.

    Requirements for wrapped env:
      - env.num_tasks: int
      - env.current_task_id: int (set on reset)
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

        assert hasattr(env, "num_tasks"), "Wrapped env must expose env.num_tasks"
        assert hasattr(env, "current_task_id"), "Wrapped env must expose env.current_task_id"
        self.num_tasks = int(env.num_tasks)

        assert isinstance(env.observation_space, gym.spaces.Box), "Only Box observation spaces are supported"

        # Extend observation space: [obs, one_hot]
        low = env.observation_space.low
        high = env.observation_space.high
        assert low.shape == high.shape

        low_aug = np.concatenate([low, np.zeros((self.num_tasks,), dtype=low.dtype)], axis=0)
        high_aug = np.concatenate([high, np.ones((self.num_tasks,), dtype=high.dtype)], axis=0)

        self.observation_space = gym.spaces.Box(low=low_aug, high=high_aug, dtype=env.observation_space.dtype)

        # Cached onehot (updated at reset)
        self._onehot = np.zeros((self.num_tasks,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        task_id = int(getattr(self.env, "current_task_id"))
        if not (0 <= task_id < self.num_tasks):
            raise ValueError(f"Invalid task_id={task_id} for num_tasks={self.num_tasks}")

        self._onehot.fill(0.0)
        self._onehot[task_id] = 1.0

        obs_aug = self.observation(obs)
        info = dict(info) if info is not None else {}
        info["task_onehot_id"] = task_id
        return obs_aug, info

    def observation(self, obs):
        # obs: (obs_dim,)
        obs = np.asarray(obs)
        return np.concatenate([obs, self._onehot], axis=0)
