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
from stable_baselines3 import TD3, DDPG, SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv




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
        # Create Meta-World MT10 environment
        # Note: Meta-World uses v3 suffix (not v2)
        env = gym.make(
            'Meta-World/MT10',
            # env_name=task_name,  laut chatgpt nicht notwendig bei MT10
            seed=seed + rank,  # Different seed for each parallel env
            reward_function_version='v3',  # Use v2 reward (default, more stable)
            max_episode_steps=max_episode_steps,  # Episode length
            terminate_on_success=False,  # Don't terminate early on success (for training)
        )

        # Optional: Normalize rewards for more stable learning
        # Uncomment if experiencing training instability
        # if normalize_reward:
        #     env = gym.wrappers.NormalizeReward(env)

        # Monitor wrapper for logging episode statistics
        # This automatically tracks episode rewards, lengths, and success rates
        env = Monitor(env)

        return env

    return _init


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    # Task Selection
    ENV_ID   = "Meta-World/MT10"
    RUN_NAME = "MT10_SAC_baseline"   # optional: "MT10_SAC", "MT10_baseline", etc.

    # Algorithm Selection
    ALGORITHM = "SAC"  # "TD3" or "DDPG" or "PPO" or "SAC" - SAC recommended for Meta-World

    # Environment Settings
    USE_PARALLEL = True  # Set to False for single environment
    N_ENVS = 10 if USE_PARALLEL else 1
    SEED = 42

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
    print(f"Meta-World MT10 Training: {TASK_NAME}")
    print(f"Algorithm: {ALGORITHM}")
    print(f"=" * 60)

    # Create vectorized training environments (parallel)
    if USE_PARALLEL:
        print(f"Creating {N_ENVS} parallel training environments...")
        env = SubprocVecEnv(
            [make_env_mt10(i, SEED, MAX_EPISODE_STEPS) for i in range(N_ENVS)],
             start_method="spawn"
       )

    else:
        print("Creating single training environment...")
        env = make_env_mt10(0, SEED, MAX_EPISODE_STEPS, NORMALIZE_REWARD)()

    # Create evaluation environment (without reward normalization for accurate eval)
    print("Creating evaluation environment...")
    eval_env = make_env_mt10(0, SEED + 1000, MAX_EPISODE_STEPS, normalize_reward=False)()

    # Get action space dimensions
    n_actions = env.action_space.shape[0]

    # Initialize the RL algorithm
    print(f"\nInitializing {ALGORITHM} agent...")

    if ALGORITHM == "TD3":
        # TD3 - Optimized for Meta-World manipulation tasks
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)  # Reduced noise for fine manipulation, 0.1 original
        )

        model = TD3(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,  # Lower LR for stability
            buffer_size=5_000_000,
            learning_starts=20_000,  # Start training sooner with parallel envs.  orig=5000
            batch_size=512,
            tau=0.001, # orig=0.005
            gamma=0.99,  # Higher gamma for longer horizon tasks
            train_freq=(1, "step"),
            gradient_steps=1,  # Train on all available data at each step, orig=-1
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.05,  # Reduced for smoother target policy, orig=0.1, best=0.05
            target_noise_clip=0.2,    # Tighter clipping, orig=0.3, best=0.2
            policy_kwargs=dict(
                net_arch=[256, 256, 256],  # Deeper network for complex policies
                activation_fn=torch.nn.ReLU,
            ),
            tensorboard_log=f"./metaworld_logs/{ALGORITHM}/",
            verbose=1,
            device="auto",
            seed=SEED,
        )

    elif ALGORITHM == "DDPG":
        # DDPG - Recommended for Meta-World (better exploration)
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma= 0.1 * np.ones(n_actions)  # Reduced noise for fine manipulation, orig=0.1, best=0.1
        )
        model = DDPG(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,  # Lower LR for stability
            buffer_size=1_000_000,
            learning_starts=5000,  # Start training sooner with parallel envs
            batch_size=256,
            tau=0.005,  # Orig=0.005
            gamma=0.99,  # Higher gamma for longer horizon tasks
            train_freq=(1, "step"),
            gradient_steps=2, # orig=-1,  Train on all available data at each step, best=2 
            action_noise=action_noise,
            policy_kwargs=dict(
                net_arch=[256, 256, 256],  # Deeper network for complex policies
                activation_fn=torch.nn.ReLU,
            ),
            tensorboard_log=f"./metaworld_logs/{ALGORITHM}/",
            verbose=1,
            device="auto",
            seed=SEED,
        )

    elif ALGORITHM == "SAC":
        # SAC - Recommended for Meta-World (better exploration)
        model = SAC(
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
            tensorboard_log=f"./metaworld_logs/{ALGORITHM}/",
            # tb_log_name="MT1_reach_SAC",
            verbose=1,
            device="auto",
            seed=SEED,
        )


    elif ALGORITHM == "PPO":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=4e-4,
            n_steps=8192,          # Timesteps pro Env vor einem Update
            batch_size=1024,            # Mini-Batch-Größe für SGD
            n_epochs=20,              # Wie oft pro Rollout über die Daten iteriert wird
            gamma=0.99,
            gae_lambda=0.95,          # GAE für Advantage-Schätzung
            clip_range=0.28,           # PPO-Clipping-Parameter ε
            clip_range_vf = None,
            normalize_advantage = False,
            ent_coef=0.01,             # ggf. 0.01 für mehr Exploration
            vf_coef=0.5,              # Gewichtung der Value-Loss-Komponente
            max_grad_norm=0.5,
            use_sde = False,
            sde_sample_freq = -1,
            tensorboard_log=f"./metaworld_logs/{ALGORITHM}/",
            policy_kwargs=dict(
                net_arch=[256, 256, 256],  # ähnlich wie bei SAC, aber etwas schlanker ok
                activation_fn=torch.nn.ReLU,
                log_std_init=-3,  # Initial exploration level
                ortho_init=True, # Enable orthogonal initialization

            ),
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
        save_path=f"./metaworld_models/checkpoints_{TASK_NAME}/",
        name_prefix=f"{ALGORITHM.lower()}_{TASK_NAME}",
        verbose=1
    )

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
        best_model_save_path=f"./metaworld_models/best_{TASK_NAME}/",
        log_path=f"./metaworld_logs/eval_{TASK_NAME}/",
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
    print(f"  - Reward function: v2 (more stable)")
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
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )

    # Save the final model
    print("\nSaving final model...")
    model.save(f"./metaworld_models/{ALGORITHM.lower()}_{TASK_NAME}_final")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: ./metaworld_models/{ALGORITHM.lower()}_{TASK_NAME}_final.zip")
    print(f"Best model saved to: ./metaworld_models/best_{TASK_NAME}/best_model.zip")
    print(f"Checkpoints saved to: ./metaworld_models/checkpoints_{TASK_NAME}/")
    print(f"\nTo monitor training, run: tensorboard --logdir=./metaworld_logs/")
    print("=" * 60)

    # Cleanup
    env.close()
    eval_env.close()
