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
from .market import IEXMarket, IEXMarketV2, AuctionMarket
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
                "quantity": Box(
                    low=-np.ones(shape=T),
                    high=+np.ones(shape=T),
                    shape=(T,),
                    dtype=np.float32,
                ),
                "price": Box(low=0.0, high=1.0, shape=(T,), dtype=np.float32),
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
                    high=self.batteries[
                        agent_id
                    ].capacity_kwh,  # TODO: scale this by some factor
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

        day_offset = np.random.randint(1, 700)
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

        Info:
        - Net energy draw from the grid
        - Cost to households for the day
        """
        self.info = {k: {} for k in self.agents}

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

        return obs, rewards, terminateds, {}, self.info

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
            agent_info = {}
            q_buy: np.ndarray = (
                action_dist[agent]["quantity"] * REGULATIONS["max_transaction_kwh"]
            )  # Scale the action to actual kWh values
            prices = (
                action_dist[agent]["price"] * REGULATIONS["price_cap"]
            )  # Scale price action to actual price range

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
            soc = []
            for t in range(REGULATIONS["time_blocks_per_day"]):
                battery_flow.append(
                    self.batteries[i].charge(net_before_battery[t])
                    if net_before_battery[t] > 0
                    else -self.batteries[i].discharge(-net_before_battery[t])
                )
                soc.append(self.batteries[i].soc)

            battery_flow = np.array(battery_flow)
            soc = np.array(soc)
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

            # Reward calculations
            R_economic = -C_dam - C_imbalance
            R_deviation = -RL_CONST["imbalance_penalty_rate"] * np.sum(np.abs(Et))
            rewards[agent] = R_economic + R_deviation

            ##### LOGGING INFO FOR ANALYSIS #####
            agent_info["Generated energy"] = np.sum(
                generation_forecast * self.agent_solar[i]
            )
            agent_info["Demand"] = np.sum(agent_demand)
            agent_info["DAM_Energy_Bought"] = np.sum(np.where(q_buy > 0, q_buy, 0))
            agent_info["DAM_Energy_Sold"] = np.sum(np.where(q_buy < 0, -q_buy, 0))
            agent_info["Net_Grid_import"] = np.sum(np.where(Et > 0, Et, 0))
            agent_info["Net_Grid_export"] = np.sum(np.where(Et < 0, -Et, 0))
            agent_info["Net_before_Battery"] = net_before_battery
            agent_info["Avg_Battery_SOC"] = np.average(soc)
            agent_info["Std_Battery_SOC"] = np.std(soc)

            agent_info["Transaction_Cost"] = C_dam
            agent_info["Imbalance_Cost"] = C_imbalance
            agent_info["Reward"] = rewards[agent]

            self.info[agent] = agent_info

        return rewards


class P2PEnergyTradingAuction(MultiAgentEnv):
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
        print(f"\n\nAgents : {self.agents} \n\n\n\n\n")
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
        self.market = IEXMarketV2()
        self.auction_logic = AuctionMarket(agents=self.agents)

        self.demand_model = [HouseholdDemand() for _ in self.agents]
        self.batteries = [Battery() for _ in self.agents]
        self.agent_solar = [
            np.random.uniform(0, self.max_solar_gen) for _ in self.agents
        ]

    def get_action_space(self, agent_id):
        T = REGULATIONS["time_blocks_per_day"]
        return Box(
            low=-np.ones(shape=2 * T),
            high=+np.ones(shape=2 * T),
            shape=(2 * T,),
            dtype=np.float32,
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
                    high=self.batteries[
                        agent_id
                    ].capacity_kwh,  # TODO: scale this by some factor
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

        day_offset = np.random.randint(1, 700)
        self.start_date = datetime(2024, 1, 1) + timedelta(days=day_offset)
        self.date = self.start_date
        self.market.reset(start_date=self.date - timedelta(days=1))

        print(self.date)

        # get prev days market price and generation forecast for obs
        self.prev_market_val = self.market.get_day_values(self.date - timedelta(days=1))
        self.curr_market_val = self.market.get_day_values(self.date)

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
                "market_price": self.prev_market_val.get("price") / 11,
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

        Info:
        - Net energy draw from the grid
        - Cost to households for the day
        """
        self.info = {k: {} for k in self.agents}

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

        return obs, rewards, terminateds, {}, self.info

    def _get_obs(self):
        """
        Observations:
        at T-1: market_price
        at T: battery_soc, battery_capacity, forecast_demand
        at T+1: generation_forecast
        """

        obs = {}
        # get prev days market price and generation forecast for obs
        self.prev_market_val = self.curr_market_val
        self.curr_market_val = self.market.get_day_values(self.date)
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
                "market_price": self.prev_market_val.get("price") / 11,
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

        # Update q_buy based on auction clearing price
        ####################################################
        quantities = {
            agent: action_dist[agent][:24] * REGULATIONS["max_transaction_kwh"]
            for agent in self.agents
        }
        prices = {
            agent: action_dist[agent][24:48] * REGULATIONS["price_cap"]
            for agent in self.agents
        }
        market_price = self.curr_market_val.get("price")
        market_volume = self.curr_market_val.get("volume")
        (q, _) = self.auction_logic.clear_auction(
            q=quantities,
            p=prices,
            market_price=market_price,
            market_volume=market_volume,
        )

        # Compute rewards based on cleared quantities q and market price
        ####################################################
        for i, agent in enumerate(self.agents):
            agent_info = {}
            q_buy: np.ndarray = q[agent]

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
            soc = []
            for t in range(REGULATIONS["time_blocks_per_day"]):
                battery_flow.append(
                    self.batteries[i].charge(net_before_battery[t])
                    if net_before_battery[t] > 0
                    else -self.batteries[i].discharge(-net_before_battery[t])
                )
                soc.append(self.batteries[i].soc)

            battery_flow = np.array(battery_flow)
            soc = np.array(soc)
            Et = (
                net_before_battery - battery_flow
            )  # Amnt to buy/sell from grid after battery flow

            C_dam = np.sum(self.curr_market_val.get("price") * q_buy)
            C_imbalance = np.sum(
                self.curr_market_val.get("price")
                * np.where(
                    Et < 0,
                    -Et * 1.5,  # buying deficit (Et negative)
                    -Et * 0.8,  # selling surplus
                )
            )

            # Reward calculations
            R_economic = -C_dam - C_imbalance
            R_deviation = -RL_CONST["imbalance_penalty_rate"] * np.sum(np.abs(Et))
            rewards[agent] = R_economic + R_deviation

            ##### LOGGING INFO FOR ANALYSIS #####
            agent_info["Market_execution_rate"] = np.sum(q_buy != 0) / len(q_buy)

            agent_info["Generated_energy"] = np.sum(
                generation_forecast * self.agent_solar[i]
            )
            agent_info["Demand"] = np.sum(agent_demand)
            agent_info["DAM_Energy_Bought"] = np.sum(np.where(q_buy > 0, q_buy, 0))
            agent_info["DAM_Energy_Sold"] = np.sum(np.where(q_buy < 0, -q_buy, 0))
            agent_info["Net_Grid_import"] = np.sum(np.where(Et > 0, Et, 0))
            agent_info["Net_Grid_export"] = np.sum(np.where(Et < 0, -Et, 0))
            agent_info["Net_before_Battery"] = net_before_battery
            agent_info["Avg_Battery_SOC"] = np.average(soc)
            agent_info["Std_Battery_SOC"] = np.std(soc)

            agent_info["Transaction_Cost"] = C_dam
            agent_info["Imbalance_Cost"] = C_imbalance
            agent_info["Reward"] = rewards[agent]

            self.info[agent] = agent_info

        return rewards
