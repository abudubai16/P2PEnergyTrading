# MARL P2P Energy Trading — Key Mapping (Concise)

**P2PEnergyEnv (RLlib MultiAgentEnv):** Orchestrates the simulation loop — gathers actions, computes PV generation and stochastic household demand, updates battery SOC, runs market clearing, calculates rewards, and returns observations.

**LoadModel:** Generates realistic household demand using daily load profiles + stochastic noise (and optional temperature sensitivity).

**PVModel:** Converts cloud cover/irradiance data into solar power output for each household based on installed capacity.

**BatteryModel:** Enforces charge/discharge limits, updates SOC dynamics, and computes battery degradation cost.

**MarketClearingEngine:** Matches buyers and sellers and computes the market clearing price and settled traded energy (P2P exchange behavior).

**TariffProvider / PriceProvider:** Supplies Indian ToD tariffs, feed‑in/net‑metering rates, and optionally IEX day‑ahead market prices.

**RLlib Policy (Agent):** Uses local observations to choose a single action (buy/store/sell decision); performs no physics, forecasting, or pricing computations.

---

## Training / Simulation Algorithm

For each episode (1 simulated day):

1. Initialize environment (SOC, time = 0, weather and tariff data loaded).
2. For each timestep (hour):
   - Each agent receives its observation.
   - Agent policy outputs an action (charge / idle / discharge).
   - Environment computes PV generation using weather data.
   - Environment generates stochastic household demand.
   - BatteryModel updates SOC and applies constraints.
   - MarketClearingEngine matches buyers and sellers and computes clearing price.
   - Tariff/price applied and rewards calculated.
   - Next observations returned to agents.

3. After 24 steps, episode ends and RLlib updates the shared policy (PPO/MAPPO).
