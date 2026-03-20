# RLlib modules
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.distribution.torch.torch_distribution import TorchDiagGaussian

# pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict


class P2PTradingPolicy(TorchRLModule, ValueFunctionAPI):
    """
    Custom RLModule for the P2P energy trading environment.

    Config options:
    - demand_hidden_size : int
    - generation_hidden_size : int
    - market_price_hidden_size: int
    - time_blocks : int (the number of time blocks per day)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override(TorchRLModule)
    def setup(self):
        """
        Observations:
        at T-1: market_price
        at T: battery_soc, battery_capacity, forecast_demand
        at T+1: generation_forecast
        """
        super().setup()
        self.action_dist_cls = TorchDiagGaussian

        self.time_blocks: int = self.model_config.get("time_blocks")
        self.demand_layer = nn.LSTM(
            input_size=1,
            hidden_size=self.model_config.get("demand_hidden_size"),
            num_layers=2,
            batch_first=True,
        )
        self.generation_layer = nn.LSTM(
            input_size=1,
            hidden_size=self.model_config.get("generation_hidden_size"),
            num_layers=2,
            batch_first=True,
        )
        self.market_price_layer = nn.LSTM(
            input_size=1,
            hidden_size=self.model_config.get("market_price_hidden_size"),
            num_layers=2,
            batch_first=True,
        )
        self.encoder_layer = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64)
        )

        decoder_input_size = (
            self.model_config.get("demand_hidden_size")
            + self.model_config.get("generation_hidden_size")
            + self.model_config.get("market_price_hidden_size")
            + 64
        )
        self.decoder_layer = nn.Sequential(
            nn.Linear(decoder_input_size, 64), nn.ReLU(), nn.Linear(64, 24)
        )
        self.value_layer = nn.Sequential(
            nn.Linear(decoder_input_size, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self.log_std = nn.Parameter(torch.zeros(24))

    def _compute_embeddings(self, batch):
        obs: Dict[str, torch.Tensor] = batch.get(Columns.OBS, batch)

        scalers = torch.cat(
            (
                obs["battery_soc"],
                obs["battery_capacity"],
            ),
            dim=-1,
        )
        scaler_embeds = self.encoder_layer(scalers)
        demand_embeds = self._return_last_timestep(
            self.demand_layer, obs["forecast_demand"]
        )
        market_price_embeds = self._return_last_timestep(
            self.market_price_layer, obs["market_price"]
        )
        generation_embeds = self._return_last_timestep(
            self.generation_layer, obs["generation"]
        )

        embeds = torch.cat(
            (
                scaler_embeds,
                demand_embeds,
                market_price_embeds,
                generation_embeds,
            ),
            dim=-1,
        )
        return embeds

    def _return_last_timestep(self, layer: nn.LSTM, obs: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = layer(obs.unsqueeze(-1))
        return h_n[-1]

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # TODO : check if the values returned will be a numpy array or tensor
        embedding = self._compute_embeddings(batch)
        logits = self.decoder_layer(embedding)
        log_std = self.log_std.expand_as(logits)
        logits = torch.cat([logits, log_std], dim=-1)

        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        embeddings = self._compute_embeddings(batch)
        logits: torch.Tensor = self.decoder_layer(embeddings)
        log_std = self.log_std.expand_as(logits)
        logits = torch.cat([logits, log_std], dim=-1)

        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: embeddings,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings = self._compute_embeddings(batch)
        value = self.value_layer(embeddings)

        return value


class P2PTradingPolicyAuction(TorchRLModule, ValueFunctionAPI):
    """
    Custom RLModule for the P2P energy trading environment.

    Config options:
    - demand_hidden_size : int
    - generation_hidden_size : int
    - market_price_hidden_size: int
    - time_blocks : int (the number of time blocks per day)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override(TorchRLModule)
    def setup(self):
        """
        Observations:
        at T-1: market_price
        at T: battery_soc, battery_capacity, forecast_demand
        at T+1: generation_forecast
        """
        super().setup()
        self.action_dist_cls = TorchDiagGaussian

        self.time_blocks: int = self.model_config.get("time_blocks")
        self.demand_layer = nn.LSTM(
            input_size=1,
            hidden_size=self.model_config.get("demand_hidden_size"),
            num_layers=2,
            batch_first=True,
        )
        self.generation_layer = nn.LSTM(
            input_size=1,
            hidden_size=self.model_config.get("generation_hidden_size"),
            num_layers=2,
            batch_first=True,
        )
        self.market_price_layer = nn.LSTM(
            input_size=1,
            hidden_size=self.model_config.get("market_price_hidden_size"),
            num_layers=2,
            batch_first=True,
        )
        self.encoder_layer = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64)
        )

        decoder_input_size = (
            self.model_config.get("demand_hidden_size")
            + self.model_config.get("generation_hidden_size")
            + self.model_config.get("market_price_hidden_size")
            + 64
        )
        self.value_layer = nn.Sequential(
            nn.Linear(decoder_input_size, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        self.q_head = nn.Sequential(
            nn.Linear(decoder_input_size, 64), nn.ReLU(), nn.Linear(64, 24)
        )
        self.p_head = nn.Sequential(
            nn.Linear(decoder_input_size, 64), nn.ReLU(), nn.Linear(64, 24)
        )

        self.q_log_std = nn.Parameter(torch.zeros(24))
        self.p_log_std = nn.Parameter(torch.zeros(24))

    def _compute_embeddings(self, batch):
        obs: Dict[str, torch.Tensor] = batch.get(Columns.OBS, batch)

        scalers = torch.cat(
            (
                obs["battery_soc"],
                obs["battery_capacity"],
            ),
            dim=-1,
        )
        scaler_embeds = self.encoder_layer(scalers)
        demand_embeds = self._return_last_timestep(
            self.demand_layer, obs["forecast_demand"]
        )
        market_price_embeds = self._return_last_timestep(
            self.market_price_layer, obs["market_price"]
        )
        generation_embeds = self._return_last_timestep(
            self.generation_layer, obs["generation"]
        )

        embeds = torch.cat(
            (
                scaler_embeds,
                demand_embeds,
                market_price_embeds,
                generation_embeds,
            ),
            dim=-1,
        )
        return embeds

    def _return_last_timestep(self, layer: nn.LSTM, obs: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = layer(obs.unsqueeze(-1))
        return h_n[-1]

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        embedding = self._compute_embeddings(batch)

        q_mean_raw = self.q_head(embedding)  # (B, 24)
        p_mean_raw = self.p_head(embedding)  # (B, 24)

        # Quantity: symmetric [-Q_max, Q_max]
        q_mean = torch.tanh(q_mean_raw)

        # Price: bounded [P_min, P_max]
        p_mean = torch.sigmoid(p_mean_raw)
        q_log_std = self.q_log_std.expand_as(q_mean)
        p_log_std = self.p_log_std.expand_as(p_mean)

        # Clamp log_std for stability
        q_log_std = torch.clamp(q_log_std, -5.0, 2.0)
        p_log_std = torch.clamp(p_log_std, -5.0, 2.0)

        mean = torch.cat([q_mean, p_mean], dim=-1)
        log_std = torch.cat([q_log_std, p_log_std], dim=-1)

        logits = torch.cat([mean, log_std], dim=-1)

        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        embeddings = self._compute_embeddings(batch)
        embedding = self._compute_embeddings(batch)

        q_mean_raw = self.q_head(embedding)  # (B, 24)
        p_mean_raw = self.p_head(embedding)  # (B, 24)

        # Quantity: symmetric [-Q_max, Q_max]
        q_mean = torch.tanh(q_mean_raw)

        # Price: bounded [P_min, P_max]
        p_mean = torch.sigmoid(p_mean_raw)
        q_log_std = self.q_log_std.expand_as(q_mean)
        p_log_std = self.p_log_std.expand_as(p_mean)

        # Clamp log_std for stability
        q_log_std = torch.clamp(q_log_std, -5.0, 2.0)
        p_log_std = torch.clamp(p_log_std, -5.0, 2.0)

        mean = torch.cat([q_mean, p_mean], dim=-1)
        log_std = torch.cat([q_log_std, p_log_std], dim=-1)

        logits = torch.cat([mean, log_std], dim=-1)

        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: embeddings,
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings = self._compute_embeddings(batch)
        value = self.value_layer(embeddings)

        return value
