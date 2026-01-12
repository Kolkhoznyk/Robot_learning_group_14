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

Extended from the original script by additional automatic hyperparameter tuning with Optuna.
"""

import os
import warnings

import gymnasium as gym
import metaworld
import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


class TrialEvalCallback(EvalCallback):
    """Callback for Optuna trial evaluation and pruning."""
    
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000, 
                 deterministic=True, verbose=0):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        # Continue with normal evaluation
        result = super()._on_step()
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Report intermediate objective value to Optuna
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            # Prune trial if performance is poor
            if self.trial.should_prune():
                self.is_pruned = True
                return False  # Stop training
        
        return result


def make_env(task_name='reach-v3', rank=0, seed=0, max_episode_steps=500, normalize_reward=False):
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
            reward_function_version='v2',  # Use v2 reward (default, more stable)
            max_episode_steps=max_episode_steps,  # Episode length
            terminate_on_success=False,  # Don't terminate early on success (for training)
            disable_env_checker=True,  # Disable passive env checker to avoid obs space warnings
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


def objective(trial: optuna.Trial, config: dict) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        config: Dictionary with training configuration
    
    Returns:
        Mean reward achieved by the agent
    """
    # ==================== SAMPLE HYPERPARAMETERS ====================
    # Learning rate: log-uniform sampling between 1e-5 and 1e-3
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    
    # Batch size: categorical choice from common values
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    
    # Discount factor (gamma): uniform sampling
    gamma = trial.suggest_float('gamma', 0.95, 0.9999)
    
    # GAE lambda: uniform sampling for advantage estimation
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
    
    # Clip range: uniform sampling for PPO clipping
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    
    # Entropy coefficient: log-uniform for exploration
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True)
    
    # Value function coefficient
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)
    
    # Network architecture: sample layer sizes
    n_layers = trial.suggest_int('n_layers', 2, 4)
    layer_size = trial.suggest_categorical('layer_size', [128, 256, 512])
    net_arch = [layer_size] * n_layers
    
    # Max gradient norm for gradient clipping
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 5.0)
    
    # Number of epochs per update
    n_epochs = trial.suggest_int('n_epochs', 5, 20)
    # ================================================================
    
    # Extract configuration
    TASK_NAME = config['task_name']
    N_ENVS = config['n_envs']
    SEED = config['seed']
    MAX_EPISODE_STEPS = config['max_episode_steps']
    NORMALIZE_REWARD = config['normalize_reward']
    TOTAL_TIMESTEPS = config['total_timesteps']
    EVAL_FREQ = config['eval_freq']
    N_EVAL_EPISODES = config['n_eval_episodes']

    # Print trial information
    print(f"\n{'='*60}")
    print(f"Optuna Trial #{trial.number}")
    print(f"{'='*60}")
    print(f"Hyperparameters:")
    print(f"  - Learning rate: {learning_rate:.2e}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Gamma: {gamma:.4f}")
    print(f"  - GAE lambda: {gae_lambda:.4f}")
    print(f"  - Clip range: {clip_range:.2f}")
    print(f"  - Entropy coef: {ent_coef:.2e}")
    print(f"  - VF coef: {vf_coef:.2f}")
    print(f"  - Network: {net_arch}")
    print(f"  - Max grad norm: {max_grad_norm:.2f}")
    print(f"  - N epochs: {n_epochs}")
    print(f"{'='*60}")
    
    # Create vectorized training environments (parallel)
    env = SubprocVecEnv(
        [make_env(TASK_NAME, i, SEED + trial.number * 1000, MAX_EPISODE_STEPS, NORMALIZE_REWARD) 
         for i in range(N_ENVS)],
        start_method='spawn'
    )
    
    # Create evaluation environment
    eval_env = make_env(TASK_NAME, 0, SEED + trial.number * 1000 + 999, MAX_EPISODE_STEPS, 
                       normalize_reward=False)()

    # Initialize PPO with Optuna-sampled hyperparameters
    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            n_epochs=n_epochs,
            policy_kwargs=dict(
                net_arch=net_arch,
                activation_fn=torch.nn.ReLU
            ),
            tensorboard_log=f"./metaworld_logs/optuna_trial_{trial.number}/",
            verbose=0,  # Reduce output during optimization
            device="cpu",  # Use CPU for MlpPolicy (GPU provides no benefit)
            seed=SEED + trial.number * 1000,
        )

        # Create Optuna callback for pruning
        eval_callback = TrialEvalCallback(
            eval_env=eval_env,
            trial=trial,
            n_eval_episodes=N_EVAL_EPISODES,
            eval_freq=EVAL_FREQ,
            deterministic=True,
            verbose=0
        )

        # Train the agent
        print(f"Training for {TOTAL_TIMESTEPS:,} timesteps...")
        
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=eval_callback,
            log_interval=10,
            progress_bar=False  # Disable to reduce clutter during optimization
        )
        
    except Exception as e:
        print(f"Trial failed with exception: {e}")
        env.close()
        eval_env.close()
        raise optuna.TrialPruned()
    
    finally:
        # Cleanup
        env.close()
        eval_env.close()

    # Check if trial was pruned
    if eval_callback.is_pruned:
        raise optuna.TrialPruned()
    
    # Return the mean reward as optimization objective
    mean_reward = eval_callback.last_mean_reward
    print(f"Trial #{trial.number} finished with mean reward: {mean_reward:.2f}")
    print("=" * 60)
    
    return mean_reward


if __name__ == "__main__":
    # ==================== CONFIGURATION ====================
    # Task Selection
    TASK_NAME = "reach-v3"  # Change to other MT1 tasks
    
    # Environment Settings
    N_ENVS = 4  # Parallel envs per trial (reduced for parallel trials)
    SEED = 42
    
    # Training Settings (for each trial)
    TOTAL_TIMESTEPS = 200_000  # Reduced for faster hyperparameter search
    MAX_EPISODE_STEPS = 500
    NORMALIZE_REWARD = False
    
    # Evaluation Settings
    EVAL_FREQ = 10000  # Evaluate every N steps
    N_EVAL_EPISODES = 10  # Episodes per evaluation
    
    # Optuna Settings
    N_TRIALS = 20  # Number of hyperparameter combinations to try
    N_JOBS = 2  # Run 2 trials in parallel (optimal for i7-9850H 6-core CPU)
    STUDY_NAME = f"ppo_{TASK_NAME}_optimization"
    # ======================================================
    
    # Create output directories
    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)
    os.makedirs("./optuna_studies", exist_ok=True)
    
    print("=" * 60)
    print("PPO Hyperparameter Optimization with Optuna")
    print(f"Task: {TASK_NAME}")
    print(f"Number of trials: {N_TRIALS}")
    print(f"Parallel jobs: {N_JOBS} (running {N_JOBS} trials simultaneously)")
    print(f"Timesteps per trial: {TOTAL_TIMESTEPS:,}")
    print(f"Estimated time: ~{(N_TRIALS // N_JOBS) * 10}-{(N_TRIALS // N_JOBS) * 15} minutes")
    print("=" * 60)
    
    # Configuration dictionary
    config = {
        'task_name': TASK_NAME,
        'n_envs': N_ENVS,
        'seed': SEED,
        'max_episode_steps': MAX_EPISODE_STEPS,
        'normalize_reward': NORMALIZE_REWARD,
        'total_timesteps': TOTAL_TIMESTEPS,
        'eval_freq': EVAL_FREQ,
        'n_eval_episodes': N_EVAL_EPISODES,
    }
    
    # Create Optuna study with median pruner
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction='maximize',  # Maximize mean reward
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage=f"sqlite:///./optuna_studies/{STUDY_NAME}.db",
        load_if_exists=True  # Resume if study exists
    )
    
    # Run optimization
    try:
        study.optimize(
            lambda trial: objective(trial, config),
            n_trials=N_TRIALS,
            n_jobs=N_JOBS,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    
    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best mean reward: {study.best_value:.2f}")
    
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        if 'learning_rate' in key or 'ent_coef' in key:
            print(f"  - {key}: {value:.2e}")
        elif isinstance(value, float):
            print(f"  - {key}: {value:.4f}")
        else:
            print(f"  - {key}: {value}")
    
    # Save best hyperparameters to file
    import json
    best_params_file = f"./optuna_studies/best_params_{TASK_NAME}.json"
    with open(best_params_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nBest hyperparameters saved to: {best_params_file}")
    
    # Optionally visualize optimization (requires plotly)
    print("\nTo visualize optimization results, use:")
    print("  import optuna")
    print(f"  study = optuna.load_study(study_name='{STUDY_NAME}', ")
    print(f"                            storage='sqlite:///./optuna_studies/{STUDY_NAME}.db')")
    print("  optuna.visualization.plot_optimization_history(study)")
    print("  optuna.visualization.plot_param_importances(study)")
    print("=" * 60)
