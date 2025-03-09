import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Battery specifications from environment.py
BATTERY_CAPACITY = 13276.4396  # kWh
C_RATE = 0.5
MAX_CHARGE_DISCHARGE_POWER = C_RATE * BATTERY_CAPACITY  # kW
SOC_MIN = 0.2 * BATTERY_CAPACITY  # 20% SoC
SOC_MAX = 0.8 * BATTERY_CAPACITY  # 80% SoC
EFFICIENCY = 0.95  # Both charging and discharging efficiency

# Hydrogen system specifications
H2_CAPACITY = 2000  # kW (2 MW)

# Initialize charging state
charging_in_progress = False
current_day = None

def get_feasible_actions(load, tou_tariff, h2_tariff, soc):
    """Determine feasible actions based on current state"""
    feasible_actions = [0]  # Do nothing is always feasible
    
    # Action 1: Battery operations (if not at extremes)
    if (soc > SOC_MIN + 1e-5) or (soc < SOC_MAX - 1e-5):
        feasible_actions.append(1)
        
    # Action 2: H2 operations (if load exists and H2 is cheaper than grid)
    if load > 0 and h2_tariff < tou_tariff:
        feasible_actions.append(2)
        
    return feasible_actions

def process_action(action, load, pv, tou_tariff, fit, h2_tariff, soc):
    """Process action and compute energy allocations"""
    allocations = {
        'pv_to_load': 0.0,
        'pv_to_battery': 0.0,
        'pv_to_grid': 0.0,
        'battery_to_load': 0.0,
        'grid_to_load': 0.0,
        'grid_to_battery': 0.0,
        'h2_to_load': 0.0
    }

    # Priority 1: Use PV for load
    allocations['pv_to_load'] = min(pv, load)
    load_remaining = load - allocations['pv_to_load']
    pv_remaining = pv - allocations['pv_to_load']

    if action in get_feasible_actions(load, tou_tariff, h2_tariff, soc):
        if action == 1:  # Battery operations
            if load_remaining > 0 and soc > SOC_MIN:
                # Discharge battery
                available_power = min(
                    MAX_CHARGE_DISCHARGE_POWER,
                    (soc - SOC_MIN) * EFFICIENCY
                )
                allocations['battery_to_load'] = min(available_power, load_remaining)
                soc -= allocations['battery_to_load'] / EFFICIENCY
                load_remaining -= allocations['battery_to_load']
            elif pv_remaining > 0 and soc < SOC_MAX:
                # Charge battery with excess PV
                available_capacity = (SOC_MAX - soc) / EFFICIENCY
                charge_power = min(MAX_CHARGE_DISCHARGE_POWER, available_capacity)
                allocations['pv_to_battery'] = min(pv_remaining, charge_power)
                soc += allocations['pv_to_battery'] * EFFICIENCY
                pv_remaining -= allocations['pv_to_battery']
                
        elif action == 2:  # H2 operations
            if load_remaining > 0 and h2_tariff < tou_tariff:
                h2_power = min(H2_CAPACITY, load_remaining)
                allocations['h2_to_load'] = h2_power
                load_remaining -= h2_power

    # Remaining PV goes to grid
    if pv_remaining > 0:
        allocations['pv_to_grid'] = pv_remaining
        
    # Remaining load met by grid
    if load_remaining > 0:
        allocations['grid_to_load'] = load_remaining

    # Ensure SoC stays within bounds
    soc = max(SOC_MIN, min(soc, SOC_MAX))

    # Calculate costs
    purchase = (allocations['grid_to_load'] + allocations['grid_to_battery']) * tou_tariff + \
              allocations['h2_to_load'] * h2_tariff
    sell = allocations['pv_to_grid'] * fit
    bill = purchase - sell

    return soc, allocations, purchase, sell, bill

def main():
    # Load model and data
    model_path = 'best_policy.h5'
    dataset_path = 'dataset.csv'
    
    model = load_model(model_path, compile=False)
    print("Model loaded successfully.")
    
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")

    # Initialize columns
    allocation_columns = [
        'pv_to_load', 'pv_to_battery', 'pv_to_grid',
        'battery_to_load', 'grid_to_load', 'grid_to_battery',
        'h2_to_load', 'Purchase', 'Sell', 'Bill', 'SoC'
    ]
    
    for col in allocation_columns:
        df[col] = np.nan

    # Initialize SoC
    soc = BATTERY_CAPACITY * 0.5  # Start at 50% SoC as in environment.py

    # Process each timestep
    print("\nProcessing dataset:")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Get current state values
        load = row['Load']
        pv = row['PV']
        tou_tariff = row['Tou Tariff']
        fit = row['FiT']
        h2_tariff = row['H2 Tariff']
        day = row['Day']
        hour = row['Hour']

        # Normalize state (using same normalization as environment.py)
        state = np.array([
            load / df['Load'].max(),
            pv / df['PV'].max(),
            tou_tariff / df['Tou Tariff'].max(),
            fit / df['FiT'].max(),
            h2_tariff / df['H2 Tariff'].max(),
            soc / BATTERY_CAPACITY,
            day / 6,  # Days 0-6
            hour / 23  # Hours 0-23
        ])

        # Get feasible actions
        feasible_actions = get_feasible_actions(load, tou_tariff, h2_tariff, soc)

        # Predict action using the model
        action_probs = model.predict(state.reshape(1, -1), verbose=0)
        mask = np.full(3, -np.inf)  # 3 possible actions
        mask[feasible_actions] = 0
        masked_probs = action_probs[0] + mask
        action = np.argmax(masked_probs)

        # Process action
        soc, allocations, purchase, sell, bill = process_action(
            action, load, pv, tou_tariff, fit, h2_tariff, soc
        )

        # Store results
        for key, value in allocations.items():
            df.at[index, key] = value
        df.at[index, 'Purchase'] = purchase
        df.at[index, 'Sell'] = sell
        df.at[index, 'Bill'] = bill
        df.at[index, 'SoC'] = (soc / BATTERY_CAPACITY) * 100  # Store as percentage

    # Save results
    output_path = 'results.csv'
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Financial metrics
    plt.subplot(2, 1, 1)
    plt.plot(df['Purchase'].rolling(24).mean(), label='Purchase')
    plt.plot(df['Sell'].rolling(24).mean(), label='Sell')
    plt.plot(df['Bill'].rolling(24).mean(), label='Net Bill')
    plt.title('24-hour Rolling Average of Financial Metrics')
    plt.legend()
    
    # Battery SoC
    plt.subplot(2, 1, 2)
    plt.plot(df['SoC'], label='Battery SoC')
    plt.axhline(y=20, color='r', linestyle='--', label='Min SoC')
    plt.axhline(y=80, color='r', linestyle='--', label='Max SoC')
    plt.title('Battery State of Charge')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('energy_management_results.png')
    plt.close()

if __name__ == "__main__":
    main()