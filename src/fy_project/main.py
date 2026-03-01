import gymnasium as gym


from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env

from fy_project.env import P2PEnergyTrading
from fy_project.agent import P2PTradingPolicy

import json
from pathlib import Path

parser = add_rllib_example_script_args(default_iters=100, default_timesteps=600000)
parser.set_defaults(
    env="P2PEnergyTrading",
    num_households=1,
)

if __name__ == "__main__":
    args = parser.parse_args()

    register_env("P2PEnergyTrading", lambda cfg: P2PEnergyTrading(**cfg))

    with open(Path("src/config/agents.json")) as f:
        agents_cfg = json.load(f)

    with open(Path("src/config/env_config.json")) as f:
        env_cfg = json.load(f)

    print("Creating experiment config...\n\n")
    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment(
            env="P2PEnergyTrading",
            env_config=env_cfg,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=P2PTradingPolicy,
                model_config=agents_cfg,
            ),
        )
    )

    print("Starting experiment... \n\n")
    run_rllib_example_script_experiment(base_config, args)
