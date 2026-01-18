"""
Meta-World MT10 Training Script with Stable Baselines3

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
import random

import gymnasium as gym
import metaworld
import numpy as np
import torch
from stable_baselines3 import TD3, DDPG, SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from metaworld_envs.mt10_env import MetaWorldMT10Env
from metaworld_envs.task_onehot_wrapper import TaskOneHotObsWrapper
from callbacks.task_metrics import MT10TaskMetricsCallback
from algo_sac_disent_alpha.sac_disentangled_alpha import SACDisentangledAlpha



def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_env_mt10(rank=0, seed=0, max_episode_steps=500, normalize_reward=False):
    """
    Create and wrap the Meta-World MT10 environment.

    Args:
        # task_name: Name of the Meta-World task (e.g., 'reach-v3', 'push-v3')
        rank: Index of the subprocess (for parallel envs)
        seed: Random seed
        max_episode_steps: Maximum steps per episode (default: 500)
        normalize_reward: Whether to normalize rewards (optional, can improve learning)
    """
    def _init():
            env = MetaWorldMT10Env(
                seed=seed + rank,
                max_episode_steps=max_episode_steps,
                terminate_on_success=False,
                task_set="train",
            )
            env = TaskOneHotObsWrapper(env)  # <-- adds one-hot to observation
            env = Monitor(env)
            return env
    return _init


if __name__ == "__main__":

    # ==================== CONFIGURATION ====================

    # Algorithm Selection
    ALGORITHM = "SAC"

    # Toggle disentangled alpha
    SAC_DISENTANGLED_ALPHA = False   # False = baseline

    # Run naming (IMPORTANT for plotting & grouping)
    RUN_GROUP = "disent_alpha" if SAC_DISENTANGLED_ALPHA else "baseline"
    RUN_NAME  = f"MT10_SAC_{RUN_GROUP}"

    # Environment settings
    USE_PARALLEL = True
    N_ENVS = 10 if USE_PARALLEL else 1
    SEED = 42

    # Training settings
    TOTAL_TIMESTEPS = 10_000        # increase later
    MAX_EPISODE_STEPS = 500
    NORMALIZE_REWARD = False

    # Evaluation / checkpointing
    EVAL_FREQ = 10_000
    N_EVAL_EPISODES = 20
    CHECKPOINT_FREQ = 25_000

    # ======================================================

    # Reproducibility
    set_global_seeds(SEED)

    # Create base output directories
    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)

    print("=" * 60)
    print(f"Meta-World MT10 Training")
    print(f"  Algorithm : {ALGORITHM}")
    print(f"  Variant   : {RUN_GROUP}")
    print(f"  Seed      : {SEED}")
    print("=" * 60)

    # ------------------------------------------------------
    # Print MT10 task-id mapping (for sanity & report)
    # ------------------------------------------------------
    _dbg = MetaWorldMT10Env(
        seed=SEED,
        max_episode_steps=MAX_EPISODE_STEPS,
        task_set="train",
    )

    print("\nMT10 task-id mapping:")
    for tid, name in enumerate(_dbg._env_names):
        print(f"  task_{tid}: {name}")
    _dbg.close()

    # ------------------------------------------------------
    # Create training environments
    # ------------------------------------------------------
    if USE_PARALLEL:
        print(f"\nCreating {N_ENVS} parallel training environments...")
        env = SubprocVecEnv(
            [
                make_env_mt10(
                    rank=i,
                    seed=SEED,
                    max_episode_steps=MAX_EPISODE_STEPS,
                    normalize_reward=NORMALIZE_REWARD,
                )
                for i in range(N_ENVS)
            ],
            start_method="spawn",
        )
    else:
        print("\nCreating single training environment...")
        env = DummyVecEnv(
            [
                make_env_mt10(
                    rank=0,
                    seed=SEED,
                    max_episode_steps=MAX_EPISODE_STEPS,
                    normalize_reward=NORMALIZE_REWARD,
                )
            ]
        )

    # ------------------------------------------------------
    # Evaluation environment (immer DummyVecEnv)
    # ------------------------------------------------------
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv(
        [
            make_env_mt10(
                rank=0,
                seed=SEED + 1000,
                max_episode_steps=MAX_EPISODE_STEPS,
                normalize_reward=False,
            )
        ]
    )

    # ------------------------------------------------------
    # Debug: observation & one-hot sanity check
    # ------------------------------------------------------
    obs = env.reset()
    print("\nAugmented obs shape:", obs.shape)
    print("Single obs dim:", env.observation_space.shape)

    num_tasks = 10
    onehot = obs[:, -num_tasks:]
    assert np.allclose(onehot.sum(axis=1), 1.0), "Invalid one-hot task encoding!"

    # ------------------------------------------------------
    # Common SAC hyperparameters
    # ------------------------------------------------------
    COMMON_SAC_ARGS = dict(
        learning_rate=3e-4,
        gamma=0.99,
        tau=5e-3,
        buffer_size=2_000_000,
        learning_starts=1500,
        batch_size=512,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        use_sde=False,
        policy_kwargs=dict(
            net_arch=[400, 400],
            activation_fn=torch.nn.ReLU,
            log_std_init=0.0,
        ),
        verbose=1,
        device="auto",
        seed=SEED,
    )

    # ------------------------------------------------------
    # Initialize model
    # ------------------------------------------------------
    print("\nInitializing SAC agent...")

    TENSORBOARD_LOGDIR = f"./metaworld_logs/{RUN_GROUP}/"

    if SAC_DISENTANGLED_ALPHA:
        model = SACDisentangledAlpha(
            "MlpPolicy",
            env,
            num_tasks=10,
            tensorboard_log="./metaworld_logs/disent_alpha",
            **COMMON_SAC_ARGS,
        )
    else:
        model = SAC(
            "MlpPolicy",
            env,
            tensorboard_log="./metaworld_logs/baseline",
            **COMMON_SAC_ARGS,
        )

    # ------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------
    task_metrics_cb = MT10TaskMetricsCallback(num_tasks=10, verbose=0)

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_{RUN_NAME}/",
        name_prefix=f"sac_{RUN_NAME}",
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./metaworld_models/best_{RUN_NAME}/",
        log_path=f"./metaworld_logs/{RUN_GROUP}/eval_seed{SEED}/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        verbose=1,
        warn=False,
    )

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------
    print(f"\nStarting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print("=" * 60)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[task_metrics_cb, checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True,
        tb_log_name=f"seed{SEED}",
    )

    # ------------------------------------------------------
    # Save final model
    # ------------------------------------------------------
    print("\nSaving final model...")
    model.save(f"./metaworld_models/sac_{RUN_NAME}_final")

    print("\nTraining complete.")
    print(f"Logs saved to:      ./metaworld_logs/{RUN_GROUP}/")
    print(f"Models saved to:    ./metaworld_models/")
    print("=" * 60)

    env.close()
    eval_env.close()





    """ Standard SAC hyperparameters for reference
    policy="MlpPolicy",
    env=env,
    learning_rate=4e-4,           # best = 3e-4
    buffer_size=2_000_000,        # Larger replay buffer for improved generalization
    learning_starts=25000,       # Begin updates later
    batch_size=512,
    tau=0.001,                   # Smaller soft-update rate for more stable target updates
    gamma=0.999,                
    train_freq=3,                 # Perform one training phase every three environment steps
    gradient_steps=3,             # Execute three gradient updates whenever training is triggered
    ent_coef='auto',              
    target_entropy='auto',        # Automatically select target policy entropy
    use_sde=True,                 # Enable state-dependent exploration for continuous control
    policy_kwargs=dict(
        net_arch=[256, 256, 256], 
        activation_fn=torch.nn.ReLU,
        log_std_init=-2,          # Lower initial exploration variance for stable early behavior
    ),
    tensorboard_log=f"./metaworld_logs/{RUN_NAME}/{ALGORITHM}/",
    # tb_log_name="MT1_reach_SAC",
    verbose=1,
    device="auto",
    seed=SEED,
    """
