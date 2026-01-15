import numpy as np
import gymnasium as gym
import metaworld


class MetaWorldMT10Env(gym.Env):
    """
    SB3-compatible single-environment wrapper that samples among MT10 tasks on reset().

    Key additions for task-conditioning:
      - Stable mapping env_name -> task_id in [0, 9]
      - Stores current_task_id and returns it in info on reset/step
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        seed: int = 0,
        max_episode_steps: int = 500,
        terminate_on_success: bool = False,
        task_set: str = "train",  # "train" or "test"
    ):
        super().__init__()
        assert task_set in ["train", "test"]

        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._max_episode_steps = int(max_episode_steps)
        self._terminate_on_success = bool(terminate_on_success)
        self._task_set = task_set

        self._benchmark = metaworld.MT10()
        self._classes = (
            self._benchmark.train_classes if task_set == "train" else self._benchmark.test_classes
        )
        self._tasks = (
            self._benchmark.train_tasks if task_set == "train" else self._benchmark.test_tasks
        )

        # Stable mapping from env_name -> task_id (0..9)
        # Important: sorted() makes mapping deterministic across runs/machines.
        self._env_names = sorted(list(self._classes.keys()))
        self._env_name_to_id = {name: i for i, name in enumerate(self._env_names)}
        self.num_tasks = len(self._env_names)  # should be 10

        # Cache of instantiated envs per env_name
        self._env_cache = {}
        self._current_env = None
        self._current_task = None
        self._elapsed_steps = 0

        # Public: current task id (0..9). Set in reset().
        self.current_task_id = None
        self.current_task_env_name = None

        # Create one env to define spaces and sanity-check consistency
        any_env_name = self._env_names[0]
        tmp_env = self._classes[any_env_name]()  # instantiate
        self.observation_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space
        tmp_env.close()

    def _get_or_create_env(self, env_name: str):
        if env_name not in self._env_cache:
            env = self._classes[env_name]()  # instantiate Mujoco env
            # Safety: ensure spaces match the wrapper spaces
            assert env.observation_space == self.observation_space, f"Obs space mismatch in {env_name}"
            assert env.action_space == self.action_space, f"Action space mismatch in {env_name}"
            self._env_cache[env_name] = env
        return self._env_cache[env_name]

    def _sample_task(self):
        idx = self._rng.integers(0, len(self._tasks))
        return self._tasks[int(idx)]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)

        self._elapsed_steps = 0

        # Random task each episode
        self._current_task = self._sample_task()
        env_name = self._current_task.env_name

        self.current_task_env_name = env_name
        self.current_task_id = int(self._env_name_to_id[env_name])

        self._current_env = self._get_or_create_env(env_name)
        self._current_env.set_task(self._current_task)

        obs, info = self._current_env.reset()
        if info is None:
            info = {}

        info = dict(info)
        info["mt_task_env_name"] = env_name
        info["mt_task_id"] = self.current_task_id

        return obs, info

    def step(self, action):
        assert self._current_env is not None, "Call reset() before step()."

        obs, reward, terminated, truncated, info = self._current_env.step(action)
        self._elapsed_steps += 1

        # Enforce max episode length at wrapper level
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        # Optional early termination on success
        if self._terminate_on_success and (info is not None) and ("success" in info) and (float(info["success"]) >= 1.0):
            terminated = True

        info = dict(info) if info is not None else {}
        info["mt_task_env_name"] = self.current_task_env_name
        info["mt_task_id"] = self.current_task_id

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._current_env is None:
            return None
        return self._current_env.render()

    def close(self):
        for env in self._env_cache.values():
            try:
                env.close()
            except Exception:
                pass
        self._env_cache.clear()
        self._current_env = None
