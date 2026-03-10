# RLlib modules
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI

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
        self.time_blocks: int = self.model_config.get("time_blocks")
        self.demand_layer = nn.LSTM(
            input_size=self.time_blocks,
            hidden_size=self.model_config.get("demand_hidden_size"),
            num_layers=2,
            batch_first=True,
        )
        self.generation_layer = nn.LSTM(
            input_size=self.time_blocks,
            hidden_size=self.model_config.get("generation_hidden_size"),
            num_layers=2,
            batch_first=True,
        )
        self.market_price_layer = nn.LSTM(
            input_size=self.time_blocks,
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
        return layer(obs)[1][0][-1].unsqueeze(0)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # TODO : check if the values returned will be a numpy array or tensor
        embedding = self._compute_embeddings(batch)
        logits = self.decoder_layer(embedding)

        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        embeddings = self._compute_embeddings(batch)
        logits: torch.Tensor = self.decoder_layer(embeddings)
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
