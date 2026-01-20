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
    
class MWEnvWrapperOneHotRewardScale(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_scales):
        super().__init__(env)
        
        assert hasattr(env, "num_tasks"), "Wrapped env must expose env.num_tasks"
        assert hasattr(env, "current_task_id"), "Wrapped env must expose env.current_task_id"
        self.num_tasks = int(env.num_tasks)
        
        self.reward_scales = reward_scales
        if reward_scales is not None:
            assert self.num_tasks is not len(self.reward_scales), "Reward scales bigger than task number!!"

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
        obs = np.asarray(obs)
        return np.concatenate([obs, self._onehot], axis=0)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.reward_scales is not None:
            reward = reward * self.reward_scales[info["mt_task_id"]]
        return self._one_hot_obs(obs), reward, terminated, truncated, info
    
    
    

    