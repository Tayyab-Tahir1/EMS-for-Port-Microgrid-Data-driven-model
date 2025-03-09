import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import DQN

# Battery specs
BATTERY_CAPACITY = 13276.4396
C_RATE = 0.5
MAX_CHARGE_DISCHARGE_POWER = C_RATE * BATTERY_CAPACITY
SOC_MIN = 0.2 * BATTERY_CAPACITY
SOC_MAX = 0.8 * BATTERY_CAPACITY
EFFICIENCY = 0.95

# Hydrogen specs
H2_CAPACITY = 2000  # kW

def get_feasible_actions(load, tou_tariff, h2_tariff, soc):
    """Determine feasible actions based on current state."""
    feasible_actions = [0]  # Action 0 (do nothing) is always feasible
    if (soc > SOC_MIN + 1e-5) or (soc < SOC_MAX - 1e-5):
        feasible_actions.append(1)  # Battery ops
    if load > 0 and h2_tariff < tou_tariff:
        feasible_actions.append(2)  # Hydrogen ops
    return feasible_actions

def process_action(action, load, pv, tou_tariff, fit, h2_tariff, soc):
    """Process the chosen action and compute energy flows."""
    allocations = {
        'pv_to_load': 0.0,
        'pv_to_battery': 0.0,
        'pv_to_grid': 0.0,
        'battery_to_load': 0.0,
        'grid_to_load': 0.0,
        'grid_to_battery': 0.0,
        'h2_to_load': 0.0
    }
    # 1) PV supplies load first
    allocations['pv_to_load'] = min(pv, load)
    load_remaining = load - allocations['pv_to_load']
    pv_remaining = pv - allocations['pv_to_load']

    # 2) Battery or hydrogen
    if action == 1:
        # Battery ops
        if load_remaining > 0 and soc > SOC_MIN:
            available_power = min(MAX_CHARGE_DISCHARGE_POWER, (soc - SOC_MIN) * EFFICIENCY)
            allocations['battery_to_load'] = min(available_power, load_remaining)
            soc -= allocations['battery_to_load'] / EFFICIENCY
            load_remaining -= allocations['battery_to_load']
        elif pv_remaining > 0 and soc < SOC_MAX:
            available_capacity = (SOC_MAX - soc) / EFFICIENCY
            charge_power = min(MAX_CHARGE_DISCHARGE_POWER, available_capacity)
            allocations['pv_to_battery'] = min(pv_remaining, charge_power)
            soc += allocations['pv_to_battery'] * EFFICIENCY
            pv_remaining -= allocations['pv_to_battery']
    elif action == 2:
        # Hydrogen ops
        if load_remaining > 0 and h2_tariff < tou_tariff:
            h2_power = min(H2_CAPACITY, load_remaining)
            allocations['h2_to_load'] = h2_power
            load_remaining -= h2_power

    # 3) Sell remaining PV
    if pv_remaining > 0:
        allocations['pv_to_grid'] = pv_remaining

    # 4) Remaining load from grid
    if load_remaining > 0:
        allocations['grid_to_load'] = load_remaining

    # Keep SoC within bounds
    soc = max(SOC_MIN, min(soc, SOC_MAX))

    # Calculate costs
    purchase = (allocations['grid_to_load'] + allocations['grid_to_battery']) * tou_tariff \
               + allocations['h2_to_load'] * h2_tariff
    sell = allocations['pv_to_grid'] * fit
    bill = purchase - sell

    return soc, allocations, purchase, sell, bill

def main():
    # 1) Load your trained Stable-Baselines3 DQN model
    model_path = "dqn_energy_model_finetuned.zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Stable-Baselines3 model not found: {model_path}")
    model = DQN.load(model_path)
    print(f"Loaded Stable-Baselines3 model from {model_path}")

    # 2) Load your dataset (CSV) with columns: 'Load', 'PV', 'Tou_Tariff', 'FiT', 'H2_Tariff', etc.
    dataset_path = 'dataset.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")

    # 3) Prepare columns in df for storing results
    allocation_columns = [
        'pv_to_load', 'pv_to_battery', 'pv_to_grid',
        'battery_to_load', 'grid_to_load', 'grid_to_battery',
        'h2_to_load', 'Purchase', 'Sell', 'Bill', 'SoC'
    ]
    for col in allocation_columns:
        df[col] = np.nan

    # Initialize battery SoC
    soc = BATTERY_CAPACITY * 0.5

    print("\nRunning validation with Stable-Baselines3 DQN model...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        load = row['Load']
        pv = row['PV']
        # Use the columns with underscores
        tou_tariff = row['Tou_Tariff']
        fit = row['FiT']
        h2_tariff = row['H2_Tariff']
        day = row['Day']
        hour = row['Hour']

        # Build normalized state
        state = np.array([
            load / df['Load'].max(),
            pv / df['PV'].max(),
            tou_tariff / df['Tou_Tariff'].max(),
            fit / df['FiT'].max(),
            h2_tariff / df['H2_Tariff'].max(),
            soc / BATTERY_CAPACITY,
            day / 6.0,
            hour / 23.0
        ], dtype=np.float32)

        # 4) Use the stable-baselines model to predict an action
        raw_action, _ = model.predict(state, deterministic=True)

        # 5) Feasibility check
        feasible_actions = get_feasible_actions(load, tou_tariff, h2_tariff, soc)
        if raw_action not in feasible_actions:
            # Fallback if chosen action is not feasible
            raw_action = np.random.choice(feasible_actions)

        # 6) Process the chosen action
        soc, allocations, purchase, sell, bill = process_action(
            raw_action, load, pv, tou_tariff, fit, h2_tariff, soc
        )

        # Store results
        for key, value in allocations.items():
            df.at[index, key] = value
        df.at[index, 'Purchase'] = purchase
        df.at[index, 'Sell'] = sell
        df.at[index, 'Bill'] = bill
        df.at[index, 'SoC'] = (soc / BATTERY_CAPACITY) * 100

    # 7) Save final results
    output_csv = 'results_stable.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # 8) Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Rolling averages for financial metrics
    plt.subplot(2, 1, 1)
    plt.plot(df['Purchase'].rolling(24).mean(), label='Purchase')
    plt.plot(df['Sell'].rolling(24).mean(), label='Sell')
    plt.plot(df['Bill'].rolling(24).mean(), label='Net Bill')
    plt.title('24-hour Rolling Average of Financial Metrics (Stable-Baselines3 DQN)')
    plt.legend()
    
    # Battery SoC over time
    plt.subplot(2, 1, 2)
    plt.plot(df['SoC'], label='Battery SoC')
    plt.axhline(y=20, color='r', linestyle='--', label='Min SoC')
    plt.axhline(y=80, color='r', linestyle='--', label='Max SoC')
    plt.title('Battery State of Charge')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('energy_management_results_stable.png')
    plt.close()
    print("Plots saved to 'energy_management_results_stable.png'.")

if __name__ == "__main__":
    main()
