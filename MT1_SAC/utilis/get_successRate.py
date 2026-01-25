import os
import re
import csv
from typing import Optional, List

import gymnasium as gym
import metaworld
import numpy as np
from stable_baselines3 import SAC



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

def main():
    TASK_NAME = "reach-v3"
    ALGORITHM = "SAC"
    N_EVAL_EPISODES = 20
    MAX_EPISODE_STEPS = 150
    SEED = 40

    CHECKPOINT_DIR = f"./metaworld_models/checkpoints_{TASK_NAME}/"
    OUTPUT_CSV = f"./analysis/MT1_{ALGORITHM}_{TASK_NAME}_SuccessRate.csv"

    if not os.path.isdir(CHECKPOINT_DIR):
        raise FileNotFoundError(f"Checkpoint dir not found: {CHECKPOINT_DIR}")
    checkpoint_files = list_checkpoint_files(CHECKPOINT_DIR)
    if not checkpoint_files:
        raise RuntimeError("No checkpoint .zip files found")
    # print(checkpoint_files)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    env = gym.make(
        'Meta-World/MT1',
        env_name=TASK_NAME,
        seed=SEED,
        # render_mode='human', # Enable visual rendering
        reward_function_version='v3',  # Use v2 reward (same as training)
        max_episode_steps=MAX_EPISODE_STEPS,  # Episode length
        terminate_on_success=True,  # Don't terminate early (for consistent evaluation)
    )

    csv_data = []

    for model_path in checkpoint_files:
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
        
        step = extract_step_from_filename(model_path)
        if ALGORITHM == "SAC":
            model = SAC.load(model_path, env=env)
        else:
            raise ValueError(f"Unknown algorithm: {ALGORITHM}")
        success_count = 0

        for epsiode in range(N_EVAL_EPISODES):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_success = False

            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)

                # Close gripper
                if TASK_NAME == "reach-v3":
                    action = action.copy()
                    action[-1] = 1.0

                obs, reward, done, truncated, info = env.step(action)

                if "success" in info and bool(info["success"]):
                    episode_success = True

            if episode_success:
                success_count += 1

        success_rate = success_count / float(N_EVAL_EPISODES)
        print(f"Model: {model_path}")
        print(f"Step: {step}")
        print(f"Success Rate: {success_rate}")
        csv_data.append([step, success_rate])

    env.close()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "success_rate",
        ])
        writer.writerows(csv_data)

    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()