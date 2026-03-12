# module imports
from fy_project.env import P2PEnergyTrading
from fy_project.agent import P2PTradingPolicy
from fy_project.callbacks import EnergyLoggerCallbacks
from fy_project.paths import CONFIG_DIR, CHECKPOINT_DIR

# internal imports
import os
import json
import logging
import warnings
from pathlib import Path

# RLlib imports
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec, MultiRLModuleSpec
from ray.rllib.algorithms.algorithm import Algorithm

# initialize ray logger
logger = logging.getLogger("ray.rllib")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def env_creator(cfg):
    return P2PEnergyTrading(**cfg)


if __name__ == "__main__":

    # Load config files
    with open(Path.joinpath(CONFIG_DIR, "env_config.json"), "r") as f:
        env_config = json.load(f)
        agent_cfg = env_config["agent_cfg"]

    # Register the env with RLlib
    register_env("P2P_env", env_creator)

    # Get the observation and action spaces from the environment
    temp_env = P2PEnergyTrading(**env_config)
    observation_space = temp_env.get_observation_space(0)
    action_space = temp_env.get_action_space(0)
    del temp_env

    # Create the RLModuleSpec for the policy
    module_spec = MultiRLModuleSpec(
        rl_module_specs={
            "shared_policy": RLModuleSpec(
                module_class=P2PTradingPolicy,
                observation_space=observation_space,
                action_space=action_space,
                model_config=agent_cfg,
            )
        }
    )

    # Create the PPOConfig
    config = (
        PPOConfig()
        .environment(
            env="P2P_env",
            env_config=env_config,
        )
        .training(train_batch_size=512)
        .callbacks(EnergyLoggerCallbacks)
        .framework("torch")
        .rl_module(rl_module_spec=module_spec)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .env_runners(num_env_runners=4, num_envs_per_env_runner=1)
        # .debugging(log_level="DEBUG")
    )

    ray.init()
    algo: Algorithm = config.build_algo()
    for _ in range(200):
        algo.train()
