import csv
from typing import Dict, List
from datetime import datetime, timedelta

from fy_project.paths import IEX_DATA_DIR, IEX_DATA_DIR2

import numpy as np


class IEXMarket:
    def __init__(self):
        self.TIME_DEL = timedelta(hours=1)
        print(IEX_DATA_DIR)
        self._buffered_row = None
        self.reset()

    def get_price(self, dt: datetime) -> float:
        try:
            if self._buffered_row is not None:
                val = self._buffered_row
                self._buffered_row = None
            else:
                val = next(self.reader)
        except StopIteration:
            self.reset()
            print("\n\n Hit end of file, looping back to start \n\n")
            if self._buffered_row is not None:
                val = self._buffered_row
                self._buffered_row = None
            else:
                val = next(self.reader)

        date_str = val["datetime"]
        if date_str != dt.strftime("%Y-%m-%d %H:%M:%S"):
            raise IndexError(
                f"Expected datetime {dt} not found in IEX data. Got {date_str} instead."
            )

        return float(val["price"]) / 1000 if "price" in val else 10.0

    def get_day_prices(self, dt: datetime) -> list[float]:
        prices = []
        for _ in range(24):
            prices.append(self.get_price(dt))
            dt += self.TIME_DEL
        return prices

    def reset(self, start_date=datetime(2024, 1, 1, 0)):
        self.reader = csv.DictReader(open(IEX_DATA_DIR, "r"))
        self._buffered_row = None

        # Skip to start_date, but keep that exact row buffered
        while True:
            try:
                val = next(self.reader)
                date_str = val["datetime"]
                if date_str == start_date.strftime("%Y-%m-%d %H:%M:%S"):
                    self._buffered_row = val  # one behind behavior for next read
                    break
            except StopIteration:
                raise ValueError(f"start_date {start_date} not found in IEX data")


class IEXMarketV2:
    """
    Indian Exchange Market data with hourly resolution having 24D vectors for the following parameters:
        -price,volume,demand_bid,supply_bid
    """

    def __init__(self):
        self.TIME_DEL = timedelta(days=1)
        print(IEX_DATA_DIR2)

    def reset(self, start_date=datetime(2024, 1, 1)) -> None:
        if start_date == datetime(2024, 1, 1):
            return

        self.reader = csv.DictReader(open(IEX_DATA_DIR2, "r"))
        self.start_date = (
            start_date - self.TIME_DEL
        )  # one behind behavior for next read

        while True:
            try:
                val = next(self.reader)
                date_str = val["date"]
                if date_str == self.start_date.strftime("%Y-%m-%d"):
                    break

            except StopIteration:
                raise ValueError(f"start_date {start_date} not found in IEX data")

    def get_day_values(self, dt: datetime) -> Dict[str, np.ndarray]:
        try:
            val = next(self.reader)

        except StopIteration:
            self.reset(self.start_date + self.TIME_DEL)
            print("\n\n Hit end of file, looping back to start \n\n")
            val = next(self.reader)

        date_str = val["date"]
        if date_str != dt.strftime("%Y-%m-%d"):
            raise IndexError(
                f"Expected datetime {dt} not found in IEX data. Got {date_str} instead."
            )
        price = val.get("price")[1:-1].split(sep=",")
        volume = val.get("volume")[1:-1].split(sep=",")
        price = np.array([float(p) for p in price], dtype=np.float32) / 1000
        volume = np.array([float(v) for v in volume], dtype=np.float32)

        return {"price": price, "volume": volume}


class AuctionMarket:
    def __init__(self, agents: List[str], num_bg_orders: int = 40, k: int = 10):
        self.num_bg_orders = num_bg_orders
        self.agents = agents
        self.k = k

    def clear_auction(
        self,
        q: Dict[str, np.ndarray],
        p: Dict[str, np.ndarray],
        market_price: np.ndarray,
        market_volume: np.ndarray,
    ) -> tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Clears a double auction with background market liquidity.

        Args:
            q: Dict[agent_id -> (24,)] quantity bids (positive=buy, negative=sell)
            p: Dict[agent_id -> (24,)] price bids
            market_price: (24,) MCP from IEX
            market_volume: (24,) cleared volume from IEX

        Returns:
            q_exec: Dict[agent_id -> (24,)] executed quantities
            clearing_price: (24,) clearing price (anchored to MCP)
        """

        T = len(market_price)

        # Output containers
        q_exec = {agent: np.zeros(T, dtype=np.float32) for agent in self.agents}

        for t in range(T):

            P_mcp = market_price[t]
            V_total = market_volume[t]

            buyers = [
                [agent, q[agent][t], p[agent][t]]
                for agent in self.agents
                if q[agent][t] > 0
            ]
            sellers = [
                [agent, -q[agent][t], p[agent][t]]
                for agent in self.agents
                if q[agent][t] < 0
            ]

            q_bg = self.k * V_total
            spread = 0.05 * P_mcp
            num_bg_orders = self.num_bg_orders  # e.g., 20–50

            # Background buyers (below MCP)
            for _ in range(num_bg_orders):
                price = np.random.uniform(P_mcp - spread, P_mcp)
                qty = q_bg / num_bg_orders
                buyers.append(["bg", qty, price])

            # Background sellers (above MCP)
            for _ in range(num_bg_orders):
                price = np.random.uniform(P_mcp, P_mcp + spread)
                qty = q_bg / num_bg_orders
                sellers.append(["bg", qty, price])

            buyers.sort(key=lambda x: -x[2])
            sellers.sort(key=lambda x: x[2])

            # Match orders
            i, j = 0, 0
            while i < len(buyers) and j < len(sellers):

                buyer_agent, buyer_q, buyer_p = buyers[i]
                seller_agent, seller_q, seller_p = sellers[j]

                # Match condition
                if buyer_p >= seller_p:

                    trade_qty = min(buyer_q, seller_q)

                    # Assign executions ONLY to real agents
                    if buyer_agent != "bg":
                        q_exec[buyer_agent][t] += trade_qty

                    if seller_agent != "bg":
                        q_exec[seller_agent][t] -= trade_qty

                    # Reduce remaining quantities
                    buyers[i][1] -= trade_qty
                    sellers[j][1] -= trade_qty

                    # Move pointers
                    if buyers[i][1] <= 1e-6:
                        i += 1
                    if sellers[j][1] <= 1e-6:
                        j += 1

                else:
                    break

        return q_exec, P_mcp
