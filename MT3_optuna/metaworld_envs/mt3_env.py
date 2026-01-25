import numpy as np
import gymnasium as gym
import metaworld

class MetaWorldMT3Env(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        task_env_names: list,
        seed: int = 0,
        max_episode_steps: int = 500,
        terminate_on_success: bool = False,
        task_set: str = "train",  # "train" or "test"
        first_task: str = "reach-v3",
    ):
        super().__init__()
        assert task_set in ["train", "test"]

        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._max_episode_steps = int(max_episode_steps)
        self._terminate_on_success = bool(terminate_on_success)
        self._task_set = task_set

        bench = metaworld.MT10()
        self._classes = bench.train_classes if task_set == "train" else bench.test_classes
        all_tasks = bench.train_tasks if task_set == "train" else bench.test_tasks

        missing = [n for n in task_env_names if n not in self._classes]
        if missing:
            raise ValueError(f"Unknown env names in MT10 {task_set}_classes: {missing}")
        
        self._tasks = [t for t in all_tasks if t.env_name in task_env_names]
        if len(self._tasks) == 0:
            raise RuntimeError(
                "No tasks found for the selected env names. "
                "Check that task_env_names match the benchmark env_name strings."
            )
        
        self._env_names = sorted(task_env_names)
        self._env_name_to_id = {name: i for i, name in enumerate(self._env_names)}
        self.num_tasks = len(task_env_names)

        self._env_cache = {}
        self._current_env = None
        self._current_task = None
        self._elapsed_steps = 0
        
        self.current_task_id = self._env_name_to_id[first_task]
        self.current_task_env_name = None
        
        tmp_env = self._classes[self._env_names[0]]()
        self.observation_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space
        tmp_env.close()

        for n in self._env_names[1:]:
            e = self._classes[n]()
            assert e.observation_space == self.observation_space, f"Obs space mismatch in {n}"
            assert e.action_space == self.action_space, f"Action space mismatch in {n}"
            e.close()

        def _get_or_create_env(self, env_name: str):
            if env_name not in self._env_cache:
                env = self._classes[env_name]()
                self._env_cache[env_name] = env
            return self._env_cache[env_name]
        
        def _sample_task(self):
            idx = (self.current_task_id+1) % 3
            return self._tasks[int(idx)]
        
        def reset(self, *, seed=None, options=None):
            if seed is not None:
                self._seed = int(seed)
                self._rng = np.random.default_rng(self._seed)

            self._elapsed_steps = 0

            self._current_task = self._sample_task()
            env_name = self._current_task.env_name

            self.current_task_env_name = env_name
            self.current_task_id = int(self._env_name_to_id[env_name])

            self._current_env = self._get_or_create_env(env_name)
            self._current_env.set_task(self._current_task)

            obs, info = self._current_env.reset()
            info = dict(info) if info is not None else {}
            info["mt_task_env_name"] = env_name
            info["mt_task_id"] = self.current_task_id
            return obs, info
        
        def step(self, action):
            assert self._current_env is not None, "Call reset() before step()."

            obs, reward, terminated, truncated, info = self._current_env.step(action)
            self._elapsed_steps += 1

            if self._elapsed_steps >= self._max_episode_steps:
                truncated = True

            if (
                self._terminate_on_success
                and info is not None
                and "success" in info
                and float(info["success"]) >= 1.0
            ):
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






