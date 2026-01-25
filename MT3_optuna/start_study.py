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
        "env_name": ["reach-v3", "push-v3", "pick-place-v3"],
        "seed": 42,
        "algorithm": "SAC",

        "reward_function_version": "v3",
        "terminate_on_success": False,

        # Parallelisierung
        "n_envs": 1,
        "use_parallel": True,
        "start_method": "spawn",

        # Evaluation
        "eval_freq": 15_000,
        "n_eval_episodes": 21, # should be dividebale by 3 for mt3!!!
        "deterministic_eval": True,

        "device": "auto",

        # Training
        "total_timesteps": 270_000,
        # "max_episode_steps": 150,

        # SAC – feste (nicht getunte) Parameter
        "policy": "MlpPolicy",
        "verbose": 1,
        
        # =========================================================
        # Optuna – Hyperparameter Search Space
        "optuna_trials": 30,
        "max_episode_steps": 150,
        "n_envs_choices": 1,

        "buffersize": (300_000, 500_000),
        "learning_starts": 10_000,

        # Learning Rate
        "lr_min": 1e-5,
        "lr_max": 5e-4,

        # Batch Sizes
        "batch_sizes": (256, 512),

        # Discount Factor
        "gamma_min": 0.92,
        "gamma_max": 0.999,

        # Target Network Update
        "tau_min": 0.0001,
        "tau_max": 0.009,
        
        "net_arch": {
            "2layer_big": [400, 400],
            "3layer_big": [256, 256, 256],
            "4layer_taper": [512, 256, 128, 64],
        },
        # =========================================================
        
        "train_freq_choices": 1,
        "gradient_steps_choices": (1, 3),  # Tupel mit einem Element!

        "target_entropy": "auto",

        "use_sde": False,

        "activation_fn": torch.nn.ReLU,
        "log_std_init": -3,
    }
    # print(params["env_name"][2])
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

    # Set first trail parameters according to mt1 push und pick-place:
    study.enqueue_trial({
        "buffersize": 300_000,
        "learning_rate": 0.0002524548054563359,
        "batch_size": 256,
        "gamma": 0.9493925038998,
        "tau": 0.004187861038711214,
        "net_key": "3layer_big",
        "gradient_steps_choices": 1,
    })
    study.enqueue_trial({
        "buffersize": 500000,
        "learning_rate": 0.0002536983386966405,
        "batch_size": 256,
        "gamma": 0.9425076029588987,
        "tau": 0.008566665643023242,
        "net_key": "3layer_big",
        "gradient_steps_choices": 1,
    })
    study.enqueue_trial({
        "buffersize": 500000,
        "learning_rate": 7.368445787171149e-05,
        "batch_size": 512,
        "gamma": 0.9381403143072642,
        "tau": 0.00032306245015071265	,
        "net_key": "4layer_taper",
        "gradient_steps_choices": 3,
    })
    study.enqueue_trial({
        "buffersize": 500000,
        "learning_rate": 5.224044313331383e-05,
        "batch_size": 256,
        "gamma": 0.9514945668624958,
        "tau": 0.00032202847572828367,
        "net_key": "4layer_taper",
        "gradient_steps_choices": 3,
    })

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

