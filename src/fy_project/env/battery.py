import numpy as np

from fy_project.paths import BATTERY_CFG


# Tested
class Battery:
    def __init__(self, timestep_hours=1.0):
        self.dt = timestep_hours
        self.efficiency = BATTERY_CFG["efficiency"]

        # -------- Randomized physical parameters --------
        self.capacity_kwh = self._sample(BATTERY_CFG["capacity_kwh"])
        self.max_charge_kw = self._sample(BATTERY_CFG["max_charge_power_kw"])
        self.max_discharge_kw = self._sample(BATTERY_CFG["max_discharge_power_kw"])

        # -------- Initial State of Charge --------
        self.soc_low, self.soc_high = BATTERY_CFG["initial_soc_range"]
        self.reset()

    def _sample(self, dist):
        val = np.random.normal(dist["mean"], dist["std"])
        return float(np.clip(val, dist["min"], dist["max"]))

    def charge(self, requested_energy_kwh):
        """
        Attempt to store energy into battery.
        Returns actual energy accepted (kWh).
        """

        # convert power limit to energy limit
        max_possible = self.max_charge_kw * self.dt

        energy = min(requested_energy_kwh, max_possible)

        # available space
        available_space = (1 - self.soc) * self.capacity_kwh

        energy = min(energy, available_space)

        # apply efficiency (loss occurs during charging)
        stored_energy = energy * self.efficiency

        self.soc += stored_energy / self.capacity_kwh
        self.soc = min(self.soc, 1.0)

        return requested_energy_kwh - energy  # energy drawn from grid/PV

    # --------------------------------------------------
    # Discharging
    # --------------------------------------------------

    def discharge(self, requested_energy_kwh):
        """
        Attempt to supply energy from battery.
        Returns actual energy delivered (kWh).
        """

        max_possible = self.max_discharge_kw * self.dt
        energy = min(requested_energy_kwh, max_possible)

        available_energy = self.soc * self.capacity_kwh
        energy = min(energy, available_energy)

        # inverter loss during discharge
        delivered_energy = energy * self.efficiency

        self.soc -= energy / self.capacity_kwh
        self.soc = max(self.soc, 0.0)

        return requested_energy_kwh - delivered_energy

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def energy_stored(self):
        return self.soc * self.capacity_kwh

    def reset(self):
        # randomize initial SOC at new episode
        self.soc = np.random.uniform(self.soc_low, self.soc_high)
