import os
import re
import csv
from typing import Optional, List

import gymnasium as gym
import metaworld
import numpy as np
from stable_baselines3 import SAC

class SingleTaskOneHotWrapper(gym.Wrapper):
    def __init__(self, env, task_id, n_tasks, reward_scale=1.0):
        super().__init__(env)
        self.task_id = task_id
        self.n_tasks = n_tasks
        self.reward_scale = reward_scale

        obs_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([obs_space.low, np.zeros(self.n_tasks)]),
            high=np.concatenate([obs_space.high, np.ones(self.n_tasks)]),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._one_hot_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._one_hot_obs(obs), reward * self.reward_scale, terminated, truncated, info

    def _one_hot_obs(self, obs):
        one_hot = np.zeros(self.n_tasks, dtype=np.float32)
        one_hot[self.task_id] = 1.0
        return np.concatenate([np.array(obs, dtype=np.float32), one_hot])

def extract_step_from_filename(fname: str) -> Optional[int]:
    step_re = re.compile(r"_(\d+)_steps\.zip$")
    m = step_re.search(fname)
    return int(m.group(1)) if m else None

def list_checkpoint_files(checkpoint_dir: str) -> List[str]:
    files = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".zip")
    ]

    def sort_key(f):
        step = extract_step_from_filename(f)
        return (0, step) if step is not None else (1, f)
    
    files.sort(key=sort_key)
    return [os.path.join(checkpoint_dir, f) for f in files]


def evaluate_model_on_env(model, env, n_eval_episodes: int):
    ep_rews = []
    successes = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        success = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

            # Single env => info ist dict
            if isinstance(info, dict) and info.get("success", 0) == 1:
                success = 1

        ep_rews.append(total_reward)
        successes.append(success)

    return float(np.mean(ep_rews)), float(np.mean(successes))


def main():
    TASK_NAMES = ["reach-v3", "push-v3", "pick-place-v3"]
    REWARD_SCALES = [0.5, 1, 1]
    ALGORITHM = "SAC"

    N_EVAL_EPISODES = 20
    MAX_EPISODE_STEPS = 150
    SEED = 30

    EXP = "MT3_1"
    CHECKPOINT_DIR = "./metaworld_models/checkpoints_MT3/"
    OUTPUT_CSV = f"./analysis/checkpoints_eval_{EXP}.csv"

    if not os.path.isdir(CHECKPOINT_DIR):
        raise FileNotFoundError(f"Checkpoint dir not found: {CHECKPOINT_DIR}")
    checkpoint_files = list_checkpoint_files(CHECKPOINT_DIR)
    if not checkpoint_files:
        raise RuntimeError("No checkpoint .zip files found")
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # --- 1) Eval-Envs EINMAL erstellen ---
    eval_envs = []
    n_tasks = len(TASK_NAMES)
    for i, task_name in enumerate(TASK_NAMES):
        env = gym.make(
            "Meta-World/MT1",
            env_name=task_name,
            seed=SEED,
            reward_function_version="v3",
            max_episode_steps=MAX_EPISODE_STEPS,
            terminate_on_success=True,   # für Success-Rate sinnvoll
        )
        env = SingleTaskOneHotWrapper(env=env, task_id=i, n_tasks=n_tasks, reward_scale=REWARD_SCALES[i])
        eval_envs.append(env)

    # CSV Header
    header = [
        "step",
        "reach mean reward", "reach successrate",
        "push mean reward", "push successrate",
        "pick-place mean reward", "pick-place successrate",
        "avg mean reward", "avg successrate",
    ]
    rows = []

     # For each checkpoint evaluate tasks!
    for model_path in checkpoint_files:
        step = extract_step_from_filename(model_path)
        if not os.path.exists(model_path):
            print(f"[skip] model missing: {model_path}")
            continue

        if ALGORITHM == "SAC":
            model = SAC.load(model_path)
        else:
            raise ValueError(f"Unknown algorithm: {ALGORITHM}")

        task_mean_rewards = []
        task_success_rates = []

        for env in eval_envs:
            mean_rew, mean_sr = evaluate_model_on_env(model, env, N_EVAL_EPISODES)
            task_mean_rewards.append(mean_rew)
            task_success_rates.append(mean_sr)

        avg_mean_rew = float(np.mean(task_mean_rewards))
        avg_mean_sr = float(np.mean(task_success_rates))

        row = [
            step,
            task_mean_rewards[0], task_success_rates[0],
            task_mean_rewards[1], task_success_rates[1],
            task_mean_rewards[2], task_success_rates[2],
            avg_mean_rew, avg_mean_sr
        ]
        rows.append(row)

        print(f"Step {step}: "
                f"reach SR={task_success_rates[0]:.2f}, "
                f"push SR={task_success_rates[1]:.2f}, "
                f"pick SR={task_success_rates[2]:.2f}, "
                f"avg SR={avg_mean_sr:.2f}")

    for env in eval_envs:
        env.close()

    # Write to csv 
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()