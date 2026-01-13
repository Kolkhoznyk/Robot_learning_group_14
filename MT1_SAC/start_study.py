import optuna
import torch
import os

from optuna_study import OptunaStudy
from pathlib import Path

def main():
    params = {
        # =========================================================
        # Meta-World / Environment
        # =========================================================
        "benchmark_id": "Meta-World/MT1",
        "env_name": "reach-v3",
        "seed": 42,
        "algorithm": "PPO",

        "reward_function_version": "v3",
        "terminate_on_success": False,

        # Parallelisierung
        "n_envs": 1,
        "use_parallel": True,
        "start_method": "spawn",

        # Evaluation
        "eval_freq": 15_000,
        "n_eval_episodes": 20,
        "deterministic_eval": True,

        "device": "auto",

        # Training
        "total_timesteps": 75_000,
        # "max_episode_steps": 150,

        # SAC – feste (nicht getunte) Parameter
        "policy": "MlpPolicy",
        "verbose": 1,
        
        # =========================================================
        # Optuna – Hyperparameter Search Space
        "optuna_trials": 30,
        "max_episode_steps": (100, 150, 200, 250),
        "n_envs_choices": (1, 4, 8, 32),

        "buffersize": (100_000, 200_000, 300_000),
        "learning_starts": 10_000,

        # Learning Rate
        "lr_min": 1e-5,
        "lr_max": 1e-3,

        # Batch Sizes
        "batch_sizes": (256, 512, 1024),

        # Discount Factor
        "gamma_min": 0.8,
        "gamma_max": 0.999,

        # Target Network Update
        "tau_min": 0.001,
        "tau_max": 0.02,
          
        "net_arch": {
            "2layer_tiny": [64, 64],
            "2layer_small": [128, 128],
            "2layer_big": [256, 256],
            "3layer_tiny": [64, 64, 64],
            "3layer_small": [128, 128, 128],
            "3layer_big": [256, 256, 256],
            "3layer_taper": [256, 128, 64],
        },
        # =========================================================
        
        "train_freq_choices": 1,
        "gradient_steps_choices": 1,  # Tupel mit einem Element!

        "target_entropy": "auto",

        "use_sde": False,

        "activation_fn": torch.nn.ReLU,
        "log_std_init": -3,
    }

    # Create log dir
    dir_path = Path(f"./optuna_db_files/{params['algorithm']}_{params['env_name']}")
    dir_path.mkdir(parents=True, exist_ok=True)
    db_path = dir_path / "optuna_study.db"
    storage = f"sqlite:///{db_path.as_posix()}"

    # Create Study Class
    optuna_runner = OptunaStudy(
        benchmark_id=params['benchmark_id'],
        env_name=params['env_name'],
        algorithm=params['algorithm'],
    )
    optuna_runner.set_params(params=params)

    # Create an empty Optuna study and start with optimze Function of Optuna class
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        optuna_runner.objective,
        n_trials=params["optuna_trials"],
        n_jobs=1,
    )

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    print("DB saved at:", os.path.abspath(db_path))

if __name__ == "__main__":
    main()

