# RL libraries
from gymnasium.spaces import Box, Dict, Text
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Math + ML libraries
import numpy as np
import torch

# Relative imports
from fy_project.paths import CONFIG_DIR
from .battery import Battery
from .demand import HouseholdDemand
from .market import IEXMarket
from .weather import Weather

# Frist party libraries
import json
from pathlib import Path
from datetime import datetime, timedelta

# https://docs.ray.io/en/latest/rllib/multi-agent-envs.html#rllib-multi-agent-environments-doc

"""
    Env-> 
    - time (0-23) - Done
    - market_price - Done
    - cloud_forecast -> generation scaled from cloud cover  - Done
    - battery_soc -> capacity + state of charge - Done
    - double auction logic 
    - Indian tariffs and other regulatory constraints and logic - Done
    - Demand prediction - Done
"""

# Indian regulations and tariffs
with open(Path.joinpath(CONFIG_DIR, "regulations.json"), "r") as f:
    REGULATIONS = json.load(f)
    DEMAND_CFG = REGULATIONS["demand_config"]
    BATTERY_CFG = REGULATIONS["battery_config"]
    DAILY_PROFILE = np.array(DEMAND_CFG["daily_profile"], dtype=np.float32)


class P2PEnergyTrading(MultiAgentEnv):
    """
    Custom env for P2P energy trading among households.

    config file expects:
    - num_households: int
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.SOLAR_CONST = 1 / 20000
        self.TIME_DEL = timedelta(days=1)
        self.max_solar_gen = kwargs.get("max_solar_generation_kwh")
        self.agents = [f"household_{i}" for i in range(kwargs["num_households"])]
        self._create_agents()
        self.EPISODE_LEN = kwargs.get("episode_length", 30)

        """
            Action space : 
                a ~ [q_buy, q_sell, q_charge, q_discharge, p_buy, p_sell]
            Observation space:
                o ~ [market_price, battery_soc, battery_capacity, forecast_demand, generation_forecast]
        """
        self.action_spaces = {
            agent: self.get_action_space(agent_id)
            for agent_id, agent in enumerate(self.agents)
        }

        self.observation_spaces = {
            agent: self.get_observation_space(agent_id)
            for agent_id, agent in enumerate(self.agents)
        }

    def _create_agents(self):
        self.weather = Weather(city="Bangalore", lat=12.9719, lon=77.5937)
        self.market = IEXMarket()

        self.demand_model = [HouseholdDemand() for _ in self.agents]
        self.batteries = [Battery() for _ in self.agents]
        self.agent_solar = [
            np.random.uniform(0, self.max_solar_gen) for _ in self.agents  # TODO
        ]

    def get_action_space(self, agent_id):
        return Dict(
            {
                # Market transactions
                "q_buy": Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array(
                        [REGULATIONS["max_transaction_kwh"]], dtype=np.float32
                    ),
                    dtype=np.float32,
                ),
                "q_sell": Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array(
                        [REGULATIONS["max_transaction_kwh"]], dtype=np.float32
                    ),
                    dtype=np.float32,
                ),
                # Battery control
                "q_charge": Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array(
                        [BATTERY_CFG["max_charge_power_kw"]["max"]], dtype=np.float32
                    ),
                    dtype=np.float32,
                ),
                "q_discharge": Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array(
                        [BATTERY_CFG["max_discharge_power_kw"]["max"]], dtype=np.float32
                    ),
                    dtype=np.float32,
                ),
                # Market prices
                "p_buy": Box(
                    low=np.array([REGULATIONS["price_floor"]], dtype=np.float32),
                    high=np.array([REGULATIONS["price_cap"]], dtype=np.float32),
                    dtype=np.float32,
                ),
                "p_sell": Box(
                    low=np.array([REGULATIONS["price_floor"]], dtype=np.float32),
                    high=np.array([REGULATIONS["price_cap"]], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

    def get_observation_space(self, agent_id):
        T = REGULATIONS["time_blocks_per_day"]
        return Dict(
            {
                # "time": Box(low=0, high=23, shape=(1,), dtype=np.int32),
                "market_price": Box(
                    low=np.zeros(T, dtype=np.float32),
                    high=np.ones(T, dtype=np.float32) + 1,
                    shape=(T,),
                    dtype=np.float32,
                ),
                "battery_soc": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "battery_capacity": Box(
                    low=0.0,
                    high=self.batteries[agent_id].capacity_kwh,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "forecast_demand": Box(
                    low=np.zeros(T, dtype=np.float32),
                    high=DAILY_PROFILE + 1,
                    shape=(T,),
                    dtype=np.float32,
                ),
                "generation": Box(
                    low=np.zeros(T, dtype=np.float32),
                    high=np.ones(T, dtype=np.float32),
                    shape=(T,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, *, seed=None, options=None):
        observations = {}

        self.date = datetime(2024, 1, 1, 0)
        self.market.reset()

        market_price = self.market.get_price(self.date)
        generation_forecast = (
            np.array(self.weather.get_day_ghi(self.date), dtype=np.float32)
            * self.SOLAR_CONST
        )

        for i, agent in enumerate(self.agents):
            observations[agent] = {
                # "time": np.array([0], dtype=np.int32),
                # "market_price": np.array([market_price], dtype=np.float32),
                "battery_soc": np.array([self.batteries[i].soc], dtype=np.float32),
                "battery_capacity": np.array(
                    [self.batteries[i].capacity_kwh], dtype=np.float32
                ),
                "forecast_demand": DAILY_PROFILE * self.demand_model[i].house_scale,
                "generation": generation_forecast * self.agent_solar[i],
            }

        return observations, {}

    def step(self, action_dist):  # TODO
        """
        Each step is one day
        One episode is 30 days (configurable)

        """

        # Check if action is valid for each agent
        for agent in self.agents:
            if self.action_spaces[agent].contains(action_dist[agent]):
                raise TypeError("Invalid action format for agent {}".format(agent))

        # Apply actions to update battery states and market interactions
        # [q_buy, q_sell, q_charge, q_discharge, p_buy, p_sell]
        for agent in self.agents:

            pass

        # Advance time and get observations
        obs = {}
        self.date += self.TIME_DEL
        t = self.date.hour

        market_price = self.market.get_price(self.date)
        generation_forecast = (
            np.array(self.weather.get_day_ghi(self.date), dtype=np.float32)
            * self.SOLAR_CONST
        )

        for agent in self.agents:
            obs[agent] = {
                "time": np.array([t], dtype=np.int32),
                "market_price": np.array([market_price], dtype=np.float32),
                "battery_soc": np.array([self.batteries[agent].soc], dtype=np.float32),
                "battery_capacity": np.array(
                    [self.batteries[agent].capacity_kwh], dtype=np.float32
                ),
                "forecast_demand": DAILY_PROFILE * self.demand_model[agent].house_scale,
                "generation": generation_forecast * self.agent_solar[agent],
            }

        # Check if the year span has been exceeded
        terminateds = {"__all__": self.date.year >= 2026}

        # Compute rewards based on actions and market interactions
        rewards = {}
        for agent in self.agents:
            action = action_dist[agent]
            # Placeholder reward logic - to be replaced with actual market and cost calculations
            rewards[agent] = -(
                action["q_buy"].item() * action["p_buy"].item()
                - action["q_sell"].item() * action["p_sell"].item()
            )

        return obs, rewards, terminateds, {}, {}
