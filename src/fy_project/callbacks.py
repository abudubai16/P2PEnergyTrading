import numpy as np

import wandb
from ray.rllib.algorithms.callbacks import DefaultCallbacks


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


# TODO - Finish implementation of this class to log metrics to wandb
class EnergyLoggerCallbacksWandb(DefaultCallbacks):

    def on_episode_start(self, *, episode, **kwargs):
        # Storage for accumulating metrics
        episode._agent_metric_buffer = {}

    def on_episode_step(self, *, episode, **kwargs):

        infos = episode.get_infos()
        if not infos:
            return

        for agent_id, agent_infos in infos.items():
            if not agent_infos:
                continue

            # Get latest info safely
            info = agent_infos[-1] if isinstance(agent_infos, list) else agent_infos

            # Initialize agent buffer if needed
            if agent_id not in episode._agent_metric_buffer:
                episode._agent_metric_buffer[agent_id] = {}

            for key, value in info.items():

                # Optional: filter only meaningful metrics
                if not isinstance(value, (int, float, np.number)):
                    continue

                if key not in episode._agent_metric_buffer[agent_id]:
                    episode._agent_metric_buffer[agent_id][key] = []

                episode._agent_metric_buffer[agent_id][key].append(value)

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):

        if not hasattr(episode, "_agent_metric_buffer"):
            return

        for agent_id, metrics in episode._agent_metric_buffer.items():

            for key, values in metrics.items():

                if len(values) == 0:
                    continue

                metric_name = f"{agent_id}/{key}"

                # Aggregate (mean over episode)
                metrics_logger.log_value(
                    metric_name,
                    float(np.mean(values)),
                    reduce="mean",
                )
