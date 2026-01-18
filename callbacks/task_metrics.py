import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class MT10TaskMetricsCallback(BaseCallback):
    """
    Log episode return and success rate per Meta-World MT10 task_id.
    Works with DummyVecEnv and SubprocVecEnv using `infos` from env.step().

    Requirements:
      - info["mt_task_id"] each step (you provide it)
      - optionally info["success"] in {0.0, 1.0}
    """

    def __init__(self, num_tasks: int = 10, verbose: int = 0, max_hist: int = 100):
        super().__init__(verbose)
        self.num_tasks = int(num_tasks)
        self.max_hist = int(max_hist)

        # Per-env episode accumulators
        self._ep_rew = None           # (n_envs,)
        self._last_task_id = None     # (n_envs,)
        self._ep_success_any = None   # (n_envs,) bool

        # Per-task history buffers
        self._task_returns = [[] for _ in range(self.num_tasks)]
        self._task_success = [[] for _ in range(self.num_tasks)]

        # sampling stats window
        self.sample_window_steps = 10_000  # VecEnv-steps (not transitions)
        self._sample_counts = None
        self._sample_total = 0
        self._window_vecenv_steps = 0

    def _init_callback(self) -> None:
        n_envs = self.training_env.num_envs
        self._ep_rew = np.zeros((n_envs,), dtype=np.float64)
        self._last_task_id = -np.ones((n_envs,), dtype=np.int64)
        self._ep_success_any = np.zeros((n_envs,), dtype=bool)

        # sampling stats window
        self._sample_counts = np.zeros((self.num_tasks,), dtype=np.int64)
        self._sample_total = 0
        self._window_vecenv_steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)
        rewards = self.locals.get("rewards", None)

        if infos is None or dones is None or rewards is None:
            return True

        # accumulate per env
        for env_idx, (info, done, r) in enumerate(zip(infos, dones, rewards)):
            self._ep_rew[env_idx] += float(r)

            tid = info.get("mt_task_id", None)
            if tid is not None:
                self._last_task_id[env_idx] = int(tid)

            if "success" in info:
                if float(info["success"]) >= 1.0:
                    self._ep_success_any[env_idx] = True

            if done:
                task_id = int(self._last_task_id[env_idx])
                if 0 <= task_id < self.num_tasks:
                    # store episode return
                    self._task_returns[task_id].append(float(self._ep_rew[env_idx]))
                    if len(self._task_returns[task_id]) > self.max_hist:
                        self._task_returns[task_id] = self._task_returns[task_id][-self.max_hist:]

                    # store episode success if available
                    if "success" in info:
                        self._task_success[task_id].append(float(self._ep_success_any[env_idx]))
                        if len(self._task_success[task_id]) > self.max_hist:
                            self._task_success[task_id] = self._task_success[task_id][-self.max_hist:]

                # reset env accumulators
                self._ep_rew[env_idx] = 0.0
                self._ep_success_any[env_idx] = False
                self._last_task_id[env_idx] = -1

        # Sampling stats
        # One VecEnv step contains n_envs transitions
        # Count how many transitions per task were sampled in the last window
        self._window_vecenv_steps += 1

        for env_idx, (info, done, r) in enumerate(zip(infos, dones, rewards)):
            self._ep_rew[env_idx] += float(r)

            task_id = info.get("mt_task_id", None)
            if task_id is not None:
                tid = int(task_id)
                self._last_task_id[env_idx] = tid

                # Count this transition towards sampling distribution
                if 0 <= tid < self.num_tasks:
                    self._sample_counts[tid] += 1
                    self._sample_total += 1

            if "success" in info:
                self._ep_success_any[env_idx] = self._ep_success_any[env_idx] or (float(info["success"]) >= 1.0)

            if done:
                tid = int(self._last_task_id[env_idx])
                if 0 <= tid < self.num_tasks:
                    self._task_returns[tid].append(float(self._ep_rew[env_idx]))
                    if "success" in info:
                        self._task_success[tid].append(float(self._ep_success_any[env_idx]))

                self._ep_rew[env_idx] = 0.0
                self._ep_success_any[env_idx] = False

        # log per-task means + mean across tasks (only tasks with data)
        task_reward_means = []
        task_success_means = []

        for k in range(self.num_tasks):
            if len(self._task_returns[k]) > 0:
                rew_mean = float(np.mean(self._task_returns[k]))
                self.logger.record(f"task/ep_rew_mean_task_{k}", rew_mean)
                task_reward_means.append(rew_mean)

            if len(self._task_success[k]) > 0:
                succ_mean = float(np.mean(self._task_success[k]))
                self.logger.record(f"task/ep_success_rate_task_{k}", succ_mean)
                task_success_means.append(succ_mean)

        if len(task_reward_means) > 0:
            self.logger.record("task/ep_rew_mean_mean", float(np.mean(task_reward_means)))

        if len(task_success_means) > 0:
            self.logger.record("task/ep_success_rate_mean", float(np.mean(task_success_means)))

    # Log sampling fractions every window steps
        # sample_window_steps counts VecEnv-steps, not transitions
        if self._window_vecenv_steps >= self.sample_window_steps and self._sample_total > 0:
            fracs = self._sample_counts.astype(np.float64) / float(self._sample_total)
            for k in range(self.num_tasks):
                self.logger.record(f"task/sample_frac_task_{k}", float(fracs[k]))

            # One scalar to summarize deviation from uniform
            uniform = 1.0 / float(self.num_tasks)
            mad = float(np.mean(np.abs(fracs - uniform)))
            self.logger.record("task/sample_frac_mean_abs_dev", mad)

            # reset window
            self._sample_counts[:] = 0
            self._sample_total = 0
            self._window_vecenv_steps = 0

        return True
