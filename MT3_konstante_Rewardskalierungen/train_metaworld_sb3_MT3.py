"""
Meta-World MT1 Training Script with Stable Baselines3

Updated to align with the latest Meta-World API:
- Uses official gymnasium.make('Meta-World/MT1', ...) registration
- Supports reward_function_version parameter (v1/v2)
- Configurable max_episode_steps
- Optional reward normalization
- Parallel training with SubprocVecEnv (spawn method)
- Comprehensive evaluation and checkpointing

For documentation, see: METAWORLD_README.md
For hyperparameter tuning guide, see: METAWORLD_TUNING.md
"""

import os
import warnings

import gymnasium as gym
import metaworld
import numpy as np
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


def make_env(task_name='reach-v3', rank=0, seed=0, max_episode_steps=500, normalize_reward=False, terminate_on_success=False):
    """
    Create and wrap the Meta-World MT1 environment.

    Args:
        task_name: Name of the Meta-World task (e.g., 'reach-v3', 'push-v3')
        rank: Index of the subprocess (for parallel envs)
        seed: Random seed
        max_episode_steps: Maximum steps per episode (default: 500)
        normalize_reward: Whether to normalize rewards (optional, can improve learning)
    """
    def _init():
        # Create Meta-World MT1 environment
        # Note: Meta-World uses v3 suffix (not v2)
        env = gym.make(
            'Meta-World/MT1',
            env_name=task_name,
            seed=seed + rank,  # Different seed for each parallel env
            reward_function_version='v3',
            max_episode_steps=max_episode_steps,  # Episode length
            terminate_on_success=terminate_on_success,  # Don't terminate early on success (for training)
        )

        # Optional: Normalize rewards for more stable learning
        # Uncomment if experiencing training instability
        if normalize_reward:
            env = gym.wrappers.NormalizeReward(env)

        # Monitor wrapper for logging episode statistics
        # This automatically tracks episode rewards, lengths, and success rates
        env = Monitor(env)

        return env

    return _init

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

        if self.reward_scales is not None:
            self.current_scale = self.reward_scales[task_id]
        else:
            self.current_scale = 1.0

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
        scaled_reward = self.current_scale * reward
        return self._one_hot_obs(obs), scaled_reward, terminated, truncated, info

    def _one_hot_obs(self, obs):
        one_hot = np.zeros(self.n_tasks, dtype=np.float32)
        one_hot[self.task_id] = 1.0
        return np.concatenate([np.array(obs, dtype=np.float32), one_hot])

class MultiTaskOneHotWrapper(gym.Env):
    """
    Wrapper to choose one environment of given task list depending on chosen probabilities.
    The environment is wrapped with SingleTaskOneHotWrapper to append the task id to the observations.
    """
    def __init__(self, envs, task_probs=None, reward_scales=None, seed=None):
        super().__init__()

        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.n_tasks = len(envs)

        if task_probs is None:
            self.task_probs = np.ones(self.n_tasks)/self.n_tasks
        else:
            self.task_probs = np.array(task_probs, dtype=np.float32)
            self.task_probs /= self.task_probs.sum()

        self.envs = [
            SingleTaskOneHotWrapper(env, task_id=i, n_tasks=self.n_tasks, reward_scales=reward_scales) for i, env in enumerate(envs)
        ]

        self.active_task_id = 0
        self.env = self.envs[0]

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        self.active_task_id = self.np_random.choice(self.n_tasks, p=self.task_probs)
        self.env = self.envs[self.active_task_id]
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

class MultiTaskEvalCallback(BaseCallback):
    """
    Evaluate for each task and save best model depending on best successrate
    """
    def __init__(self, eval_envs, n_eval_episodes=10, eval_freq=10000, save_path=None):
        super().__init__()
        self.eval_envs = eval_envs
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_sr = -np.inf
        self.save_path = save_path

    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            all_tasks_rewards = []
            all_tasks_success_rates = []
            for task_name, env in self.eval_envs.items():
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

                # Prints for each task
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

            # Mean success rate over all tasks (to save best model)
            mean_all_tasks_success_rates = np.average(all_tasks_success_rates, weights=[1, 1, 1])
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

def make_multitask_env(task_list, rank=0, seed=0, max_episode_steps=500, normalize_reward=False, task_probs=None, reward_scales=None):
    """
    Choose one of the tasks in task_list with given probabilites
    """
    def _init():
        envs = [make_env(task, rank, seed+i*100, max_episode_steps=max_episode_steps,
                          normalize_reward=normalize_reward)() for i, task in enumerate(task_list)]
        env = MultiTaskOneHotWrapper(envs, task_probs=task_probs, reward_scales=reward_scales, seed=seed)
        return env
    
    return _init


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================

    # MT3
    TASK_LIST = ["reach-v3", "push-v3", "pick-place-v3"]
    TASK_PROBS = None
    REWARD_SCALES = [0.1, 0.8, 1.0]

    # Algorithm Selection
    ALGORITHM = "SAC"  # SAC or PPO

    # Environment Settings
    USE_PARALLEL = True  # Set to False for single environment
    N_ENVS = 32 if USE_PARALLEL else 1
    SEED = 42
    # Training Settings
    TOTAL_TIMESTEPS = 2_000_000  # Increased for better convergence
    MAX_EPISODE_STEPS = 500  # Maximum steps per episode
    NORMALIZE_REWARD = True  # Set to True if experiencing training instability

    # Evaluation Settings
    EVAL_FREQ = 10000  # Evaluate every N steps
    N_EVAL_EPISODES = 20  # Number of episodes for evaluation
    CHECKPOINT_FREQ = 25000  # Save checkpoint every N steps
    # ======================================================

    # Create output directories
    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)

    print(f"=" * 60)
    print(f"Meta-World MT3 Training: {TASK_LIST}")
    print(f"Algorithm: {ALGORITHM}")
    print(f"=" * 60)

    # Create vectorized training environments (parallel)

    print(f"Creating {N_ENVS} parallel training environments...")
    env = SubprocVecEnv(
        [make_multitask_env(TASK_LIST, rank=i, seed=SEED, max_episode_steps=MAX_EPISODE_STEPS, normalize_reward=NORMALIZE_REWARD, task_probs=TASK_PROBS, reward_scales=REWARD_SCALES) for i in range(N_ENVS)],
        start_method='spawn'
    )

    # Create evaluation environments (without reward normalization for accurate eval)
    print("Creating evaluation environments...")
    eval_envs = {
        task: SingleTaskOneHotWrapper(
            make_env(task, rank=0, seed=SEED+1000, max_episode_steps=MAX_EPISODE_STEPS, normalize_reward=False, terminate_on_success=True)(),
            task_id=i,
            n_tasks=len(TASK_LIST),
            reward_scales=None
        )
        for i, task in enumerate(TASK_LIST)
    }

    # Get action space dimensions
    n_actions = env.action_space.shape[0]

    # Initialize the RL algorithm
    print(f"\nInitializing {ALGORITHM} agent...")

    if ALGORITHM == "SAC":
        # SAC - Recommended for Meta-World (better exploration)
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.000252,
            buffer_size=1_500_000,
            learning_starts=5000,  # Start training sooner
            batch_size=1024,
            tau=0.0041,
            gamma=0.97,  # Higher gamma for multi-step tasks
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
            device="cuda",
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

    # Callbacks
    # Save checkpoint every CHECKPOINT_FREQ steps
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_MT3/",
        name_prefix=f"{ALGORITHM.lower()}_MT3",
        verbose=1
    )

    multi_eval_cb = MultiTaskEvalCallback(
        eval_envs=eval_envs,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=EVAL_FREQ,
        save_path=f"./metaworld_models/best_MT3_model.zip"
    )

    # Train the agent
    total_timesteps = TOTAL_TIMESTEPS
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("=" * 60)
    print("Training configuration:")
    print(f"  - Tasks: {TASK_LIST}")
    print(f"  - Algorithm: {ALGORITHM}")
    print(f"  - Parallel environments: {N_ENVS}")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Batch size: {model.batch_size}")
    print(f"  - Gamma: {model.gamma}")
    print(f"  - Learning starts: {model.learning_starts}")
    print(f"  - Buffer size: {model.buffer_size:,}")
    print(f"  - Network architecture: [512, 512, 256]")
    print(f"  - Gradient steps: -1 (train on all data)")
    print(f"  - Seed: {SEED}")
    print(f"  - Max episode steps: {MAX_EPISODE_STEPS}")
    print(f"  - Reward function: v3")
    print(f"  - Normalize reward: {NORMALIZE_REWARD}")
    print(f"  - Eval frequency: {EVAL_FREQ} steps")
    print(f"  - Eval episodes: {N_EVAL_EPISODES}")
    print(f"  - Checkpoint frequency: {CHECKPOINT_FREQ} steps")
    
    if ALGORITHM == "SAC":
        print(f"  - Entropy tuning: Automatic")
        print(f"  - Target entropy: Automatic")
    print("=" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, multi_eval_cb],
        log_interval=10,
        progress_bar=True
    )

    # Save the final model
    print("\nSaving final model...")
    model.save(f"./metaworld_models/{ALGORITHM.lower()}_MT3_final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: ./metaworld_models/{ALGORITHM.lower()}_MT3_final.zip")
    print(f"Best model saved to: ./metaworld_models/best_MT3_model.zip")
    print(f"Checkpoints saved to: ./metaworld_models/checkpoints_MT3/")
    print(f"\nTo monitor training, run: tensorboard --logdir=./metaworld_logs/")
    print("=" * 60)

    # Cleanup
    env.close()

    for e in eval_envs.values():
        e.close()
