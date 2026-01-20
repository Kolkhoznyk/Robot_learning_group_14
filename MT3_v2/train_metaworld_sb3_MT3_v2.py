import os

import gymnasium as gym
import metaworld
import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

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

def make_env(task_name, task_id, n_tasks, rew_scale, rank, seed, max_steps, terminate_on_success=False):
    def _init():
        env = gym.make(
            'Meta-World/MT1',
            env_name=task_name,
            seed=seed + rank,
            reward_function_version='v3',
            max_episode_steps=max_steps,
            terminate_on_success=terminate_on_success
        )
        env = Monitor(env)
        return SingleTaskOneHotWrapper(env, task_id, n_tasks, rew_scale)
    return _init

class MultiTaskEvalCallback(BaseCallback):
    def __init__(self, unique_tasks, n_eval_episodes=10, eval_freq=10000, save_path=None, seed=42, max_steps=500, terminate_on_success=True):
        super().__init__()
        self.unique_tasks = unique_tasks
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_sr = -np.inf
        self.save_path = save_path
        self.seed = seed
        self.max_steps = max_steps
        self.terminate_on_success = terminate_on_success

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            all_tasks_rewards = []
            all_tasks_success_rates = []
            n_tasks = len(self.unique_tasks)

            for i, task_name in enumerate(self.unique_tasks):
                eval_env_fn = make_env(task_name, i, n_tasks, 1.0, 999, self.seed, self.max_steps, terminate_on_success=self.terminate_on_success)
                env = eval_env_fn()

                ep_rews, ep_lens, successes = [], [], []

                for _ in range(self.n_eval_episodes):
                    obs, _ = env.reset()
                    done = False
                    total_reward = 0.0
                    length = 0
                    success = 0

                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        total_reward += reward
                        length += 1
                        
                        if isinstance(info, list):
                            if info[0].get('success', 0) == 1:
                                success = 1
                        else:
                            if info.get('success', 0) == 1:
                                success = 1

                    ep_rews.append(total_reward)
                    ep_lens.append(length)
                    successes.append(success)

                mean_rew = np.mean(ep_rews)
                mean_len = np.mean(ep_lens)
                mean_success = np.mean(successes)
                all_tasks_rewards.append(mean_rew)
                all_tasks_success_rates.append(mean_success)

                print(f"Task: {task_name}")
                print("-" * 30)
                print("| rollout/           |          |")
                print(f"|    ep_len_mean     | {mean_len:.0f}      |")
                print(f"|    ep_rew_mean     | {mean_rew:.2f}    |")
                print(f"|    success_mean    | {mean_success:.2f}    |")
                print("-" * 30)

                self.logger.record(f"eval/{task_name}/ep_rew_mean", mean_rew)
                self.logger.record(f"eval/{task_name}/ep_len_mean", mean_len)
                self.logger.record(f"eval/{task_name}/success_rate", mean_success)
                
                env.close()

            mean_all_tasks_success_rates = np.mean(all_tasks_success_rates)
            mean_all_tasks_rewards = np.mean(all_tasks_rewards)

            print(f"|    ep_rew_mean of all tasks     | {mean_all_tasks_rewards:.2f}    |")
            print(f"|    success_mean of all tasks    | {mean_all_tasks_success_rates:.2f}    |")

            self.logger.record("eval/mean_ep_rew_all_tasks", mean_all_tasks_rewards)
            self.logger.record("eval/mean_success_rates_all_tasks", mean_all_tasks_success_rates)

            if self.save_path is not None and mean_all_tasks_success_rates > self.best_mean_sr:
                self.best_mean_sr = mean_all_tasks_success_rates
                self.model.save(self.save_path) 
                print(f"New best model saved with mean success rate over all tasks = {mean_all_tasks_success_rates:.2f}")
                print(f"Mean ep_rew of all tasks = {mean_all_tasks_rewards:.2f}")

            self.logger.dump(self.num_timesteps)
            
        return True

if __name__ == "__main__":
    TRAINING_TASKS = (
        ["reach-v3"] * 2 + 
        ["push-v3"] * 3 + 
        ["pick-place-v3"] * 3
    )
    print(TRAINING_TASKS)
    
    UNIQUE_TASKS = list(dict.fromkeys(TRAINING_TASKS))
    REWARD_SCALES = {"reach-v3": 0.1, "push-v3": 1.0, "pick-place-v3": 1.0}
    
    ALGORITHM = "SAC"  # SAC or PPO
    SEED = 42
    TOTAL_TIMESTEPS = 1_500_000
    MAX_EPISODE_STEPS = 500
    EVAL_FREQ = 10000
    N_EVAL_EPISODES = 10
    CHECKPOINT_FREQ = 50000

    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)

    print(f"Creating {len(TRAINING_TASKS)} parallel environments...")
    env_fns = [
        make_env(name, UNIQUE_TASKS.index(name), len(UNIQUE_TASKS), REWARD_SCALES[name], i, SEED, MAX_EPISODE_STEPS, terminate_on_success=False)
        for i, name in enumerate(TRAINING_TASKS)
    ]
    env = SubprocVecEnv(env_fns, start_method='spawn')

    if ALGORITHM == "SAC":
        # SAC - Recommended for Meta-World (better exploration)
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=2e-4,
            buffer_size=250_000,
            learning_starts=5000,  # Start training sooner
            batch_size=256,
            tau=0.005,
            gamma=0.99,  # Higher gamma for multi-step tasks
            train_freq=1,
            gradient_steps=-1,  # Train on all available data
            ent_coef='auto',  # Automatic entropy tuning - crucial for SAC
            target_entropy='auto',  # Automatically set target entropy
            use_sde=False,  # State-dependent exploration (can be enabled for more exploration)
            policy_kwargs=dict(
                net_arch=[256, 256, 256],  # Deeper network
                activation_fn=torch.nn.ReLU,
                log_std_init=-3,  # Initial exploration level
            ),
            tensorboard_log=f"./metaworld_logs/{ALGORITHM}/",
            verbose=1,
            device="auto",
            seed=SEED,
        )
    elif ALGORITHM == "PPO":
        # PPO
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,  # Higher gamma for multi-step tasks
            gae_lambda = 0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            policy_kwargs=dict(
                net_arch=[256, 256, 256],  # Deeper network
                activation_fn=torch.nn.ReLU,
            ),
            tensorboard_log=f"./metaworld_logs/{ALGORITHM}/",
            verbose=1,
            device="auto",
            seed=SEED,
        )
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_MT3/",
        name_prefix=f"{ALGORITHM.lower()}_MT3",
        verbose=1
    )

    multi_eval_cb = MultiTaskEvalCallback(
        UNIQUE_TASKS,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        save_path=f"./metaworld_models/best_MT3_model.zip",
        seed=SEED,
        max_steps=MAX_EPISODE_STEPS,
        terminate_on_success=True
    )

    model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, multi_eval_cb],
            log_interval=10,
            progress_bar=True
        )

    model.save(f"./metaworld_models/{ALGORITHM.lower()}_MT3_final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: ./metaworld_models/{ALGORITHM.lower()}_MT3_final.zip")
    print(f"Best model saved to: ./metaworld_models/best_MT3_model.zip")
    print(f"Checkpoints saved to: ./metaworld_models/checkpoints_MT3/")
    print(f"\nTo monitor training, run: tensorboard --logdir=./metaworld_logs/")
    print("=" * 60)

    env.close()