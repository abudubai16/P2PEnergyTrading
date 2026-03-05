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
    RL_CONST = REGULATIONS["rl_const"]


class P2PEnergyTrading(MultiAgentEnv):
    """
    Custom env for P2P energy trading among households.

    config file expects:
    - num_households: int
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.SOLAR_CONST: float = 1 / 20000
        self.max_solar_gen: float = kwargs.get("max_solar_generation_kwh")
        self.EPISODE_LEN: int = kwargs.get("episode_length", 30)

        self.TIME_DEL = timedelta(days=1)
        self.agents = [f"household_{i}" for i in range(kwargs["num_households"])]
        self._create_agents()

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
            np.random.uniform(0, self.max_solar_gen) for _ in self.agents
        ]

    def get_action_space(self, agent_id):
        T = REGULATIONS["time_blocks_per_day"]
        return Dict(
            {
                "q_buy": Box(
                    low=-np.ones(shape=T) - 1,
                    high=+np.ones(shape=T) + 1,
                    shape=(T,),
                    dtype=np.float32,
                ),
            }
        )

    def get_observation_space(self, agent_id):
        T = REGULATIONS["time_blocks_per_day"]
        return Dict(
            {
                "market_price": Box(
                    low=np.zeros(T, dtype=np.float32),
                    high=np.ones(T, dtype=np.float32),
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
                    high=DAILY_PROFILE,
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
        """
        Observations:
        at T-1: market_price
        at T: battery_soc, battery_capacity, forecast_demand
        at T+1: generation_forecast
        """
        obs = {}

        day_offset = np.random.randint(1, 701)
        self.start_date = datetime(2024, 1, 1) + timedelta(days=day_offset)
        self.date = self.start_date
        self.market.reset(start_date=self.date - timedelta(days=1))

        print(self.date)

        # get prev days market price and generation forecast for obs
        self.prev_market_price = self.market.get_day_prices(
            self.date - timedelta(days=1)
        )
        self.curr_market_price = self.market.get_day_prices(self.date)

        # Next days generation forecast based on cloud cover
        self.generation_forecast = (
            np.array(
                self.weather.get_day_ghi(self.date + timedelta(days=1)),
                dtype=np.float32,
            )
            * self.SOLAR_CONST
        )

        for i, agent in enumerate(self.agents):
            obs[agent] = {
                "market_price": np.array(self.prev_market_price, dtype=np.float32) / 11,
                "battery_soc": np.array([self.batteries[i].soc], dtype=np.float32),
                "battery_capacity": np.array(
                    [self.batteries[i].capacity_kwh], dtype=np.float32
                ),
                "forecast_demand": DAILY_PROFILE * self.demand_model[i].house_scale / 2,
                "generation": self.generation_forecast * self.agent_solar[i],
            }

        return obs, {}

    def step(self, action_dist):
        """
        Each step is one day
        One episode is 30 days (configurable)
        """
        # Check if action is valid for each agent
        self._action_check_validity(action_dist)

        # Advance the time, t=T+1
        self.date += self.TIME_DEL

        # Advance time and get observations
        obs = self._get_obs()

        # [q_buy, q_sell, q_charge, q_discharge, p_buy, p_sell]
        rewards = self._compute_rewards(action_dist)

        # Check if the year span has been exceeded
        terminateds = {"__all__": (self.date - self.start_date).days > self.EPISODE_LEN}

        return obs, rewards, terminateds, {}, {}

    def _get_obs(self):
        """
        Observations:
        at T-1: market_price
        at T: battery_soc, battery_capacity, forecast_demand
        at T+1: generation_forecast
        """

        obs = {}
        # get prev days market price and generation forecast for obs
        self.prev_market_price = self.curr_market_price
        self.curr_market_price = self.market.get_day_prices(self.date)
        # Next days generation forecast based on cloud cover
        self.generation_forecast = (
            np.array(
                self.weather.get_day_ghi(self.date + timedelta(days=1)),
                dtype=np.float32,
            )
            * self.SOLAR_CONST
        )

        for i, agent in enumerate(self.agents):
            obs[agent] = {
                "market_price": np.array(self.prev_market_price, dtype=np.float32) / 11,
                "battery_soc": np.array([self.batteries[i].soc], dtype=np.float32),
                "battery_capacity": np.array(
                    [self.batteries[i].capacity_kwh], dtype=np.float32
                ),
                "forecast_demand": DAILY_PROFILE * self.demand_model[i].house_scale / 2,
                "generation": self.generation_forecast * self.agent_solar[i],
            }

        return obs

    def _action_check_validity(self, action_dist):
        for agent in self.agents:
            if not self.action_spaces[agent].contains(action_dist[agent]):
                print(f"Invalid action for agent {agent}: {action_dist[agent]}")
                raise TypeError(f"Invalid action format for agent {agent}")

    def _compute_rewards(self, action_dist: Dict):
        """
        Rewards:
        at T = 0 we have prev_market_price from t-1, generation_forecast for t+1, and demand forecast for t
        """
        rewards = {}
        prev_day = self.date - timedelta(days=2)

        # Get the prev days generation for reward calculation
        generation_forecast = (
            np.array(
                self.weather.get_day_ghi(self.date),
                dtype=np.float32,
            )
            * self.SOLAR_CONST
        )

        for i, agent in enumerate(self.agents):
            q_buy: np.ndarray = action_dist[agent]["q_buy"]
            agent_demand = self.demand_model[i].get_daily_profile(
                day_of_week=prev_day.weekday(),
                day_of_year=prev_day.timetuple().tm_yday,
            )

            # From load balancing equation: generation + q_buy = demand + q_charge_battery
            net_before_battery = (
                generation_forecast * self.agent_solar[i] - agent_demand + q_buy
            )

            # Charge the batteries and calculate the remaining leftover energy that needs to be bought/sold from the grid
            battery_flow = []
            for t in range(REGULATIONS["time_blocks_per_day"]):
                battery_flow.append(
                    self.batteries[i].charge(net_before_battery[t])
                    if net_before_battery[t] > 0
                    else -self.batteries[i].discharge(-net_before_battery[t])
                )

            battery_flow = np.array(battery_flow)
            Et = (
                net_before_battery - battery_flow
            )  # Amnt to buy/sell from grid after battery flow

            C_dam = np.sum(self.curr_market_price * q_buy)
            C_imbalance = np.sum(
                self.curr_market_price
                * np.where(
                    Et < 0,
                    -Et * 1.5,  # buying deficit (Et negative)
                    -Et * 0.8,  # selling surplus
                )
            )
            R_economic = -C_dam - C_imbalance

            R_deviation = -RL_CONST["imbalance_penalty_rate"] * np.sum(np.abs(Et))
            rewards[agent] = R_economic + R_deviation

        return rewards
