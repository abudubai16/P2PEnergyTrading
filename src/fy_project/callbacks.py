import numpy as np
import pandas as pd

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

import wandb

from dotenv import load_dotenv

load_dotenv()

import os


class EnergyLoggerCallbacks(DefaultCallbacks):

    def on_episode_step(self, *, episode, metrics_logger, **kwargs):

        infos = episode.get_infos()

        for agent_id, agent_infos in infos.items():
            if not agent_infos:
                continue

            info = agent_infos[-1]

            for key, value in info.items():
                if "household" not in key:
                    continue

                metric_name = f"{agent_id}/{key}"
                metrics_logger.log_value(metric_name, value, reduce="mean")


class EnergyLoggerCallbacksWandb(DefaultCallbacks):

    def on_episode_start(self, *, episode, **kwargs):
        # Storage for accumulating metrics
        self._agent_metric_buffer = {}

    # -------- STEP COLLECTION -------- #
    def on_episode_step(self, *, episode: MultiAgentEpisode, **kwargs):

        infos = episode.get_infos()
        if not infos:
            return

        for agent_id, agent_infos in infos.items():
            if not agent_infos:
                continue

            info = agent_infos[-1] if isinstance(agent_infos, list) else agent_infos

            # Initialize agent buffer if needed
            if agent_id not in self._agent_metric_buffer:
                self._agent_metric_buffer[agent_id] = {}

            for key, value in info.items():

                # Optional: filter only meaningful metrics
                if not isinstance(value, (int, float, np.number)):
                    continue

                if key not in self._agent_metric_buffer[agent_id]:
                    self._agent_metric_buffer[agent_id][key] = []

                self._agent_metric_buffer[agent_id][key].append(value)

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):

        if not hasattr(self, "_agent_metric_buffer"):
            return

        for agent_id, metrics in self._agent_metric_buffer.items():
            for key, values in metrics.items():
                if len(values) == 0 or "household" not in key:
                    continue

                metric_name = f"{agent_id}/{key}"

                metrics_logger.log_value(
                    metric_name,
                    float(np.mean(values)),
                    reduce="mean",
                )


class EnergyLoggerCallbacksWandbV2(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.MEAN_KEYS = [
            "Market_execution_rate",
            "Generated_energy",
            "Demand",
            "DAM_Energy_Bought",
            "DAM_Energy_Sold",
            "Net_Grid_import",
            "Net_Grid_export",
            "Net_before_Battery",
            "Avg_Battery_SOC",
            "Std_Battery_SOC",
            "Reward",
            "Q_Buy",
            "Q_Sell",
            "Buying_diff",
            "Selling_diff",
            "R_DAM",
            "R_Imbalance",
            "R_Competitiveness",
            "R_Battery",
            "Reward",
        ]
        self.SUM_KEYS = []

    def on_algorithm_init(
        self, *, algorithm, metrics_logger: MetricsLogger = None, **kwargs
    ):
        """Called once when the algorithm is set up — init WandB here."""
        if wandb.run is None:
            wandb.init(
                project=os.getenv("PROJECT_NAME"),
                entity=os.getenv("ENTITY"),
                name="ppo-run-training-final",
                mode="online",
                reinit=True,
                config=algorithm.config.to_dict(),
                sync_tensorboard=True,
            )
            print("\n\n\nWandB initialized:", wandb.run.url, "\n\n\n")
        else:
            print("\n\n\nWandB already running:", wandb.run.url, "\n\n\n")

    def on_episode_step(
        self, *, episode: MultiAgentEpisode, env_runner, env, env_index, **kwargs
    ):
        """Collect per-step info from all agents."""
        # episode.get_infos() returns list of {agent_id: info_dict} per step
        infos = episode.get_infos()
        for agent_id, agent_infos in infos.items():
            if agent_id not in episode.custom_data.keys():
                episode.custom_data[agent_id] = {}

            if not agent_infos:
                continue

            info = agent_infos[-1]
            for key, value in info.items():
                if np.isnan(value) or value is None:
                    print(
                        f"Skipping logging for {agent_id} - {key} due to NaN or None value"
                    )
                    continue
                if key not in episode.custom_data[agent_id]:
                    episode.custom_data[agent_id][key] = []

                episode.custom_data[agent_id][key].append(value)

    def on_episode_end(
        self,
        *,
        episode: MultiAgentEpisode,
        env_runner,
        env,
        env_index,
        metrics_logger: MetricsLogger = None,
        **kwargs,
    ):
        # Log all scalar values
        for agent, info in episode.custom_data.items():
            for key, values in info.items():
                if values is None or len(values) == 0:
                    print(
                        f"Skipping logging for {agent} - {key} due to empty or None values"
                    )
                    continue
                if key in self.MEAN_KEYS:
                    metrics_logger.log_value(
                        f"{agent}/{key}", np.mean(values), window=1
                    )
                if key in self.SUM_KEYS:
                    metrics_logger.log_value(
                        f"{agent}/{key}_sum", np.sum(values), window=1
                    )

    def on_train_result(self, *, algorithm, metrics_logger=None, result, **kwargs):
        if wandb.run is None:
            print("WARNING: wandb.run is None in on_train_result, skipping log")
            return

        # Grab custom metrics from result
        env_runner_metrics = result.get("env_runners", {})
        # print(
        #     f"Logging env_runner metrics to WandB: {env_runner_metrics} \n\n {env_runner_metrics.keys()}"
        # )

        custom_metrics = {
            k: v
            for k, v in env_runner_metrics.items()
            if isinstance(v, (int, float, np.float32))
        }

        log_dict = {**custom_metrics}
        log_dict["training_iteration"] = result.get("training_iteration", 0)

        wandb.log(log_dict)

    def __delete__(self):
        """Called when the algorithm is cleaned up — finish the WandB run."""
        if wandb.run is not None:
            print("\n\n\nFinishing WandB run:", wandb.run.url, "\n\n\n")
            wandb.finish()
