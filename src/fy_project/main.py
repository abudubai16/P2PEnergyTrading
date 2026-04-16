# module imports
from fy_project.env import (
    P2PEnergyTrading,
    P2PEnergyTradingAuction,
    get_action_space,
    get_observation_space,
)
from fy_project.agent import (
    P2PTradingPolicy,
    P2PTradingPolicyAuction,
    P2PTradingPolicyAuctionV2,
)
from fy_project.callbacks import (
    EnergyLoggerCallbacksWandb,
    EnergyLoggerCallbacksWandbV2,
)
from fy_project.paths import ENV_CFG, AGENT_CFG, REGULATIONS, CHECKPOINT_DIR

# internal imports
import os
import logging
import warnings
from dotenv import load_dotenv

load_dotenv()

# RLlib imports
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module import RLModuleSpec, MultiRLModuleSpec, RLModule

# Ray Tune imports
from ray.tune import Tuner, TuneConfig
from ray.air import RunConfig, CheckpointConfig
from ray.air.integrations.wandb import WandbLoggerCallback

# initialize ray logger
logger = logging.getLogger("ray.rllib")
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["WANDB_MODE"] = "online"

CPU_COUNT = os.cpu_count()


def train(
    env_class=P2PEnergyTrading,
    policy_class: RLModule = P2PTradingPolicy,
    debug: bool = False,
    use_tuner: bool = False,
):
    env_name = "P2P_env_auction" if env_class == P2PEnergyTradingAuction else "P2P_env"
    num_iterations = 200

    # Get the observation and action spaces from the environment
    observation_space = get_observation_space(REGULATIONS["time_blocks_per_day"])
    action_space = get_action_space(REGULATIONS["time_blocks_per_day"])

    # Create the RLModuleSpec for the policy
    module_spec = MultiRLModuleSpec(
        rl_module_specs={
            "shared_policy": RLModuleSpec(
                module_class=policy_class,
                observation_space=observation_space,
                action_space=action_space,
                model_config=AGENT_CFG,
            )
        }
    )
    # )

    # Create the PPOConfig
    config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config=ENV_CFG,
        )
        .training(train_batch_size=2048, num_epochs=5)
        .callbacks(EnergyLoggerCallbacksWandbV2)
        .framework("torch")
        .rl_module(rl_module_spec=module_spec)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .env_runners(num_env_runners=CPU_COUNT - 3, num_envs_per_env_runner=1)
        .fault_tolerance(
            restart_failed_env_runners=True,
            delay_between_env_runner_restarts_s=5,
            num_consecutive_env_runner_failures_tolerance=10,
            restart_failed_sub_environments=True,
        )
        .api_stack(
            enable_env_runner_and_connector_v2=True, enable_rl_module_and_learner=True
        )
    )

    if debug:
        config.training(train_batch_size=128, num_epochs=2)
        config.env_runners(num_env_runners=1, num_envs_per_env_runner=1)
        config.debugging(log_level="DEBUG")
        num_iterations = 2

    if use_tuner:
        ray.init(
            _system_config={
                "gcs_rpc_server_reconnect_timeout_s": 300,
                "gcs_server_request_timeout_seconds": 300,
                "worker_register_timeout_seconds": 300,
            }
        )

        tuner = Tuner(
            "PPO",
            param_space=config,
            tune_config=TuneConfig(num_samples=1),
            run_config=RunConfig(
                name=os.getenv("PROJECT_NAME"),
                storage_path=rf"{CHECKPOINT_DIR}",
                stop={"training_iteration": num_iterations},
                callbacks=[],
                checkpoint_config=CheckpointConfig(
                    num_to_keep=3,
                    checkpoint_at_end=True,
                    checkpoint_frequency=10,
                    checkpoint_score_attribute="env_runners/episode_reward_mean",
                    checkpoint_score_order="max",
                ),
                verbose=1,
            ),
        )

        results = tuner.fit()
        best_result = results.get_best_result()
        best_checkpoint = best_result.checkpoint
        print(best_checkpoint)

    else:
        ray.init()
        algo: Algorithm = config.build_algo()
        for _ in range(200):
            algo.train()


if __name__ == "__main__":
    train(
        env_class=P2PEnergyTradingAuction,
        policy_class=P2PTradingPolicyAuctionV2,
        debug=False,
        use_tuner=True,
    )
