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
from algos.sac_disentangled_alpha import SACDisentangledAlpha



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
    ALGORITHM = "SAC"  #

    # ENHANCDED MT-SAC WITH DISENTANGLED ALPHAS
    SAC_DISENTANGLED_ALPHA  = True  # baseline: False

    # Task Selection
    ENV_ID   = "Meta-World/MT10"
    RUN_NAME = "MT10_SAC_" + ("disent_alpha" if SAC_DISENTANGLED_ALPHA else "baseline")

    # Environment Settings
    USE_PARALLEL = True  # Set to False for single environment
    N_ENVS = 10 if USE_PARALLEL else 1
    SEED = 42

    # Reproducibility through fixed seeds
    set_global_seeds(SEED)

    # Training Settings
    TOTAL_TIMESTEPS = 5_000_000 # orig=4_000_000  # Increased for better convergence
    MAX_EPISODE_STEPS = 500  # Maximum steps per episode
    NORMALIZE_REWARD = False  # Set to True if experiencing training instability

    # Evaluation Settings
    EVAL_FREQ = 10000  # Evaluate every N steps
    N_EVAL_EPISODES = 20  # Number of episodes for evaluation
    CHECKPOINT_FREQ = 25000  # Save checkpoint every N steps
    # ======================================================

    # Create output directories
    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)

    print(f"=" * 60)
    print(f"Meta-World MT10 Training: {RUN_NAME}")
    print(f"Algorithm: {ALGORITHM}")
    print(f"=" * 60)

    # Print MT10 task-id mapping (main process, independent of VecEnv type)
    _dbg = MetaWorldMT10Env(seed=SEED, max_episode_steps=MAX_EPISODE_STEPS, task_set="train")
    print("\nMT10 task-id mapping:")
    for tid, name in enumerate(_dbg._env_names):
        print(f"  task_{tid}: {name}")
    _dbg.close()


    # Create vectorized training environments (parallel)
    if USE_PARALLEL:
        print(f"Creating {N_ENVS} parallel training environments...")
        env = SubprocVecEnv(
            [make_env_mt10(rank=i, seed=SEED, max_episode_steps=MAX_EPISODE_STEPS, normalize_reward=NORMALIZE_REWARD)
            for i in range(N_ENVS)],
            start_method="spawn" 
       )

    else:
        print("Creating single training environment...")
        env = DummyVecEnv([make_env_mt10(rank=0, seed=SEED, max_episode_steps=MAX_EPISODE_STEPS, normalize_reward=NORMALIZE_REWARD)])

    # Create evaluation environment (without reward normalization for accurate eval)
    print("Creating evaluation environment...")
    # eval_env = make_env_mt10(0, SEED + 1000, MAX_EPISODE_STEPS, normalize_reward=False)() old
    eval_env = DummyVecEnv([make_env_mt10(rank=0, seed=SEED+1000, max_episode_steps=MAX_EPISODE_STEPS, normalize_reward=False)])

    # Get action space dimensions
    n_actions = env.action_space.shape[0]

    # Initialize the RL algorithm
    print(f"\nInitializing {ALGORITHM} agent...")

    # DEBUG: check obs shape
    obs = env.reset()
    print("Augmented obs shape:", obs.shape)  # VecEnv: (n_envs, obs_dim+10)
    print("Single obs dim:", env.observation_space.shape)  # should be (orig_dim+10,)
    #print(env.observation_space)

    # DEBUG: Subproc-safe sanity check for one-hot correctness
    num_tasks = 10  # MT10
    onehot = obs[:, -num_tasks:]          # last 10 dims are one-hot
    sums = onehot.sum(axis=1)             # sum per env
    print("onehot sums per env:", sums)
    assert np.allclose(sums, 1.0), "One-hot vector should have exactly one 1 per env!"

    # DEBUG: verify one-hot changes across resets
    if isinstance(env, DummyVecEnv):
        for k in range(7):
            obs = env.reset()

            base = env.envs[0]   # Monitor
            mw = base.env.env    # MetaWorldMT10Env

            tid = mw.current_task_id
            num_tasks = mw.num_tasks

            onehot = obs[0, -num_tasks:]
            print(f"reset {k}: task_id={tid}, argmax={onehot.argmax()}, sum={onehot.sum():.1f}")

            assert abs(onehot.sum() - 1.0) < 1e-6
            assert int(onehot.argmax()) == int(tid)

    # Gemeinsame Parameter von Baseline-SAC und Disentangled-Alpha-SAC
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


    if SAC_DISENTANGLED_ALPHA:
        model = SACDisentangledAlpha("MlpPolicy", env, num_tasks=10,
                                    tensorboard_log=f"./metaworld_logs/{RUN_NAME}/SAC_DISENT_ALPHA/",
                                    **COMMON_SAC_ARGS)
    elif ALGORITHM == "SAC":
        model = SAC("MlpPolicy", env,
                    tensorboard_log=f"./metaworld_logs/{RUN_NAME}/SAC_PAPER/",
                    **COMMON_SAC_ARGS)

    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")
    


    # CALLBACKS
    # Save checkpoint every CHECKPOINT_FREQ steps
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_{RUN_NAME}/",
        name_prefix=f"{ALGORITHM.lower()}_{RUN_NAME}",
        verbose=1
    )

    task_metrics_cb = MT10TaskMetricsCallback(num_tasks=10, verbose=0)

    """
    # Training automatisch stoppen, sobald Reward gut genug
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=700000,   # <- Ziel-Reward hier anpassen
        verbose=1
    )
    """

    # Evaluate every EVAL_FREQ steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./metaworld_models/best_{RUN_NAME}/",
        log_path=f"./metaworld_logs/eval_{RUN_NAME}/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
        # callback_on_new_best=stop_callback,   # <- HIER WIRD DER STOP-CALLBACK AKTIV
        verbose=1,
        warn=False
    )

    # Train the agent
    total_timesteps = TOTAL_TIMESTEPS
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("=" * 60)
    print("Training configuration:")
    print(f"Meta-World Training: {RUN_NAME}")
    print(f"  - Algorithm: {ALGORITHM}")
    print(f"  - Parallel environments: {N_ENVS}")
    if ALGORITHM in ["SAC", "TD3", "DDPG"]:
        print(f"  - Learning starts: {model.learning_starts}")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Batch size: {model.batch_size}")
    print(f"  - Gamma: {model.gamma}")
    if ALGORITHM in ["SAC", "TD3", "DDPG"]:
        print(f"  - Learning starts: {model.learning_starts}")
    if ALGORITHM in ["SAC", "TD3", "DDPG"]:
        print(f"  - Buffer size: {model.buffer_size:,}")
    print(f"  - Network architecture: [256, 256, 256]")
    print(f"  - Gradient steps: -1 (train on all data)")
    print(f"  - Seed: {SEED}")
    print(f"  - Max episode steps: {MAX_EPISODE_STEPS}")
    # print(f"  - Reward function: v2 (more stable)")
    print(f"  - Normalize reward: {NORMALIZE_REWARD}")
    print(f"  - Eval frequency: {EVAL_FREQ} steps")
    print(f"  - Eval episodes: {N_EVAL_EPISODES}")
    print(f"  - Checkpoint frequency: {CHECKPOINT_FREQ} steps")
    if ALGORITHM == "TD3":
        print(f"  - Exploration noise: σ=0.1")
        print(f"  - Target policy noise: 0.1 (clip: 0.3)")
    elif ALGORITHM == "SAC":
        print(f"  - Entropy tuning: Automatic")
        print(f"  - Target entropy: Automatic")
    print("=" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[task_metrics_cb, checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )

    # Save the final model
    print("\nSaving final model...")
    model.save(f"./metaworld_models/{ALGORITHM.lower()}_{RUN_NAME}_final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: ./metaworld_models/{ALGORITHM.lower()}_{RUN_NAME}_final.zip")
    print(f"Best model saved to: ./metaworld_models/best_{RUN_NAME}/best_model.zip")
    print(f"Checkpoints saved to: ./metaworld_models/checkpoints_{RUN_NAME}/")
    print(f"\nTo monitor training, run: tensorboard --logdir=./metaworld_logs/")
    print("=" * 60)

    # Cleanup
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
