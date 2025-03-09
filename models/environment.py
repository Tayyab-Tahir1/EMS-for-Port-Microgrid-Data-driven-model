import numpy as np
import pandas as pd

class EnergyEnv:
    def __init__(self, data):
        self.data = data.reset_index(drop=True)
        self.max_steps = len(data)
        
        # Battery specifications
        self.battery_capacity = 13276.4396  # kWh
        self.c_rate = 0.5
        self.max_charge_discharge_power = self.c_rate * self.battery_capacity  # kW
        self.soc_min = 0.2 * self.battery_capacity  # 20% SoC
        self.soc_max = 0.8 * self.battery_capacity  # 80% SoC
        self.efficiency = 0.95  # Both charging and discharging efficiency

        # Hydrogen system specifications
        self.h2_capacity = 2000  # kW (2 MW)
        
        # Calculate maximum possible power in the system for normalization
        self.max_power = max(
            data['Load'].max(),
            data['PV'].max(),
            self.max_charge_discharge_power,
            self.h2_capacity
        )
        
        # Store maximum values for normalization
        self.load_max = data['Load'].max()
        self.pv_max = data['PV'].max()
        self.tou_max = data['Tou_Tariff'].max()
        self.fit_max = data['FiT'].max()
        self.h2_max = data['H2_Tariff'].max()

        print("System Capacities:")
        print(f"Battery Capacity: {self.battery_capacity:.2f} kWh")
        print(f"Battery Power: {self.max_charge_discharge_power:.2f} kW")
        print(f"H2 System Capacity: {self.h2_capacity:.2f} kW")
        print("\nMaximum Values:")
        print(f"Max Power: {self.max_power:.2f} kW")
        print(f"Max Load: {self.load_max:.2f} kW")
        print(f"Max PV: {self.pv_max:.2f} kW")
        print(f"Max ToU Tariff: {self.tou_max:.4f}")
        print(f"Max FiT: {self.fit_max:.4f}")
        print(f"Max H2 Tariff: {self.h2_max:.4f}")

        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.total_reward = 0
        self.done = False
        self.soc = self.battery_capacity * 0.5  # Start at 50% SoC
        return self._get_state()
    
    def _get_state(self):
        row = self.data.iloc[self.current_step]
        
        state = np.array([
            row['Load'] / self.max_power,
            row['PV'] / self.max_power,
            row['Tou_Tariff'] / self.tou_max,
            row['FiT'] / self.fit_max,
            row['H2_Tariff'] / self.h2_max,
            self.soc / self.battery_capacity,
            row['Day'] / 6,
            row['Hour'] / 23
        ])
        return state

    def get_feasible_actions(self):
        row = self.data.iloc[self.current_step]
        load = row['Load']
        tou_tariff = row['Tou_Tariff']
        h2_tariff = row['H2_Tariff']
        
        feasible_actions = [0]  # Do nothing is always feasible
        
        # Action 1: Battery operations (if not at extremes)
        if (self.soc > self.soc_min + 1e-5) or (self.soc < self.soc_max - 1e-5):
            feasible_actions.append(1)
            
        # Action 2: H2 operations (if load exists and H2 is cheaper than grid)
        if load > 0 and h2_tariff < tou_tariff:
            feasible_actions.append(2)
            
        return feasible_actions

    def step(self, action):
        if self.done:
            info = {
                'pv_to_load': 0.0,
                'pv_to_battery': 0.0,
                'pv_to_grid': 0.0,
                'battery_to_load': 0.0,
                'grid_to_load': 0.0,
                'grid_to_battery': 0.0,
                'h2_to_load': 0.0,
                'Purchase': 0.0,
                'Sell': 0.0,
                'Bill': 0.0,
                'SoC': self.soc / self.battery_capacity * 100,
                'System_Max_Power': self.max_power
            }
            return np.zeros(8), 0, self.done, info

        try:
            row = self.data.iloc[self.current_step]
        except IndexError:
            print(f"Error: current_step {self.current_step} is out of bounds.")
            return np.zeros(8), 0, True, {}

        # Get current values
        load = row['Load']
        pv = row['PV']
        tou_tariff = row['Tou_Tariff']
        fit = row['FiT']
        h2_tariff = row['H2_Tariff']
        
        # Initialize energy flow variables
        pv_to_load = min(pv, load)
        load_remaining = load - pv_to_load
        pv_remaining = pv - pv_to_load
        
        pv_to_battery = 0.0
        pv_to_grid = 0.0
        battery_to_load = 0.0
        grid_to_load = 0.0
        grid_to_battery = 0.0
        h2_to_load = 0.0

        # Process actions
        if action in self.get_feasible_actions():
            if action == 1:  # Battery operations
                if load_remaining > 0 and self.soc > self.soc_min:
                    # Discharge battery
                    available_power = min(
                        self.max_charge_discharge_power,
                        (self.soc - self.soc_min) * self.efficiency
                    )
                    battery_to_load = min(available_power, load_remaining)
                    self.soc -= battery_to_load / self.efficiency
                    load_remaining -= battery_to_load
                elif pv_remaining > 0 and self.soc < self.soc_max:
                    # Charge battery with excess PV
                    available_capacity = (self.soc_max - self.soc) / self.efficiency
                    charge_power = min(self.max_charge_discharge_power, available_capacity)
                    pv_to_battery = min(pv_remaining, charge_power)
                    self.soc += pv_to_battery * self.efficiency
                    pv_remaining -= pv_to_battery
                    
            elif action == 2:  # H2 operations
                if load_remaining > 0 and h2_tariff < tou_tariff:
                    h2_power = min(self.h2_capacity, load_remaining)
                    h2_to_load = h2_power
                    load_remaining -= h2_power
        else:
            # Invalid action penalty
            reward = -10
            self.current_step += 1
            self.done = self.current_step >= self.max_steps
            next_state = self._get_state() if not self.done else np.zeros(8)
            info = {
                'pv_to_load': 0.0,
                'pv_to_battery': 0.0,
                'pv_to_grid': 0.0,
                'battery_to_load': 0.0,
                'grid_to_load': 0.0,
                'grid_to_battery': 0.0,
                'h2_to_load': 0.0,
                'Purchase': 0.0,
                'Sell': 0.0,
                'Bill': 0.0,
                'SoC': self.soc / self.battery_capacity * 100,
                'System_Max_Power': self.max_power
            }
            return next_state, reward, self.done, info

        # Remaining PV goes to grid if FiT > 0
        if pv_remaining > 0 and fit > 0:
            pv_to_grid = pv_remaining
            
        # Remaining load met by grid
        if load_remaining > 0:
            grid_to_load = load_remaining

        # Ensure SoC stays within bounds
        self.soc = max(self.soc_min, min(self.soc, self.soc_max))

        # Calculate costs and normalize them
        grid_cost = (grid_to_load + grid_to_battery) * tou_tariff
        h2_cost = h2_to_load * h2_tariff
        pv_revenue = pv_to_grid * fit
        
        # Total bill
        bill = grid_cost + h2_cost - pv_revenue
        
        # Calculate reward (negative normalized bill)
        max_possible_bill = self.max_power * max(self.tou_max, self.h2_max)
        reward = -bill / max_possible_bill

        # Additional penalty for SoC violation (shouldn't happen due to constraints)
        if self.soc < self.soc_min - 1e-5 or self.soc > self.soc_max + 1e-5:
            reward -= 5

        self.total_reward += reward
        self.current_step += 1
        self.done = self.current_step >= self.max_steps

        next_state = self._get_state() if not self.done else np.zeros(8)

        info = {
            'pv_to_load': pv_to_load,
            'pv_to_battery': pv_to_battery,
            'pv_to_grid': pv_to_grid,
            'battery_to_load': battery_to_load,
            'grid_to_load': grid_to_load,
            'grid_to_battery': grid_to_battery,
            'h2_to_load': h2_to_load,
            'Purchase': grid_cost + h2_cost,
            'Sell': pv_revenue,
            'Bill': bill,
            'SoC': self.soc / self.battery_capacity * 100,
            'System_Max_Power': self.max_power
        }

        return next_state, reward, self.done, info

    def state_size(self):
        return 8