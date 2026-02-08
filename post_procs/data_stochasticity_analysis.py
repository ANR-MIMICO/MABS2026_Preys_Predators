import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import itertools as it
import openturns as ot  # Requires: pip install openturns
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from matplotlib import rcParams
from smt.sampling_methods import LHS
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor, plot_tree, export_text
import matplotlib.pyplot as plt
rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})


# --- 3. Load & Preprocess Data ---
input_filename = 'final_aggregated_results_with_regime_v1.csv'
df = pd.read_csv(input_filename)
excluded_columns = [
    'filename', 'Tick', 'Bandicoots', 'Invaders', 'Foxes',
    'Grass_Health_Index', 'Total_Bandicoots_Eaten', 'Trap_Deaths',
    'Old_Age_Deaths', 'Hunger_Deaths', 'regime', 
    'IF', 'IG', 'IR', 'IH', 'IV'
]
features = [col for col in df.columns 
            if col not in excluded_columns 
            and pd.api.types.is_numeric_dtype(df[col])]

X_orig = df[features].values
y_orig = df['regime']
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X_orig)
y_scaled =  scaler_y.fit_transform(np.array(y_orig).reshape(-1, 1))[:, 0]



import pandas as pd

# 1. Define your input parameters (X) 
# We exclude 'S' (Seed) and the target 'y' (Target_Regime)
feature_cols = [
    'PT', 'Gr', 'PM', 'PR', 'BF', 
    'BG', 'BR', 'BH', 'BV', 'FG', 
    'FR', 'FH', 'FV'
]

# 2. Group by these columns and aggregate the target 'y' into a list
# This assumes 'df' is your DataFrame with 3250 rows
grouped_data = df.groupby(feature_cols)['regime'].agg(list).reset_index()

# 3. Rename the column for clarity
grouped_data.rename(columns={'regime': 'y_vectors'}, inplace=True)

# 4. Verify the result
# The shape should be (650, 14) -> 13 parameter columns + 1 column of lists
print(f"New shape: {grouped_data.shape}") 

# Check if every list has exactly 5 elements
counts = grouped_data['y_vectors'].apply(len)
print(f"All groups have 5 runs: {counts.unique() == [5]}")

# Example of what the output looks like:
#    PT   Gr   ...  y_vectors
# 0  10.0 55.0 ...  [0.0, 0.0, 0.5, 0.0, 0.0]
# 1  12.5 30.0 ...  [1.0, 1.0, 1.0, 1.0, 1.0]


import numpy as np

# ... [Your previous code creating 'grouped_data'] ...

# 5. Calculate the Median for each group (The "Oracle" Prediction)
# We use numpy to find the median of the list of 5 outcomes
grouped_data['group_median'] = grouped_data['y_vectors'].apply(np.median)

# 6. Calculate Performance Metrics
# Function to count how many seeds in the list match the median
def count_exact_matches(row):
    # Returns count (0 to 5) of runs that are identical to the median
    return sum([1 for y in row['y_vectors'] if y == row['group_median']])

# Function to calculate Sum of Squared Errors (SSE) for the group
def calculate_group_sse(row):
    # Returns sum of squared differences between raw runs and the median
    return sum([(y - row['group_median'])**2 for y in row['y_vectors']])

# Apply functions row-by-row
grouped_data['match_count'] = grouped_data.apply(count_exact_matches, axis=1)
grouped_data['sse'] = grouped_data.apply(calculate_group_sse, axis=1)

# 7. Compute Global Aggregates
total_runs = grouped_data['match_count'].sum() # Should be 3250 if we count matches, but we divide by total runs (3250)
total_possible_runs = len(grouped_data) * 5 # 650 * 5 = 3250

# Best Possible Accuracy: (Total Exact Matches) / (Total Runs)
theoretical_accuracy = grouped_data['match_count'].sum() / total_possible_runs

# Minimum Possible MSE: (Total Sum of Squared Errors) / (Total Runs)
theoretical_mse = grouped_data['sse'].sum() / total_possible_runs

# 8. Print Results
print("-" * 40)
print(f"THEORETICAL LIMITS (Aleatoric Uncertainty)")
print("-" * 40)
print(f"Total Configurations: {len(grouped_data)}")
print(f"Total Simulation Runs: {total_possible_runs}")
print(f"Best Possible Accuracy: {theoretical_accuracy:.2%}")
print(f"Minimum Possible MSE:   {theoretical_mse:.5f}")
print("-" * 40)

# Optional: Inspect the 'noisiest' configurations
# These are the places where the median does a poor job (high variance)
grouped_data['variance'] = grouped_data['sse'] / 5
print("\nTop 5 Noisiest Parameter Sets (Hardest to Predict):")
print(grouped_data.sort_values('variance', ascending=False).head(5)[['PT', 'BG', 'y_vectors', 'variance']])


import pandas as pd
import numpy as np

# [Assume 'grouped_data' is already created as in the previous step]
# grouped_data columns: [Parameters..., 'y_vectors']

# 1. Function to find Mode and count its frequency
def calculate_mode_accuracy(y_list):
    # pandas.mode() returns all modes in case of a tie (e.g., [0, 0, 1, 1, 0.5])
    # We take the count of the first mode found, as tied modes have the same frequency.
    modes = pd.Series(y_list).mode()
    best_mode = modes[0] 
    
    # How many times does this mode appear in the 5 runs?
    matches = y_list.count(best_mode)
    return matches

# 2. Apply to every group
grouped_data['mode_matches'] = grouped_data['y_vectors'].apply(calculate_mode_accuracy)

# 3. Calculate Global Statistics
total_correct_predictions = grouped_data['mode_matches'].sum()
total_runs = len(grouped_data) * 5  # 3250

best_possible_accuracy = total_correct_predictions / total_runs

print("-" * 40)
print(f"THEORETICAL LIMIT (Classification / Mode)")
print("-" * 40)
print(f"Best Possible Accuracy: {best_possible_accuracy:.2%}")
print("-" * 40)

# Optional: Inspect 'Confused' States (where the mode is weak)
# If the list is [0, 0.5, 1, 0, 1], the mode count is 2 (only 40% reliable)
grouped_data['consistency'] = grouped_data['mode_matches'] / 5
print("\nMost Chaotic Configurations (Low Consistency):")
print(grouped_data.sort_values('consistency').head(5)[['PT', 'y_vectors', 'consistency']])


import numpy as np
import pandas as pd

# [Assuming 'grouped_data' is already created from previous steps]
# If not, recreate it:
# grouped_data = df.groupby(feature_cols)['regime'].agg(list).reset_index()
# grouped_data.rename(columns={'regime': 'y_vectors'}, inplace=True)

# 1. Calculate the Median for each group
grouped_data['group_median'] = grouped_data['y_vectors'].apply(np.median)

# 2. Count mismatches (instances != median)
def count_mismatches(row):
    # Iterate through the 5 runs in the list
    # Count how many are NOT equal to the median
    return sum([1 for y in row['y_vectors'] if y != row['group_median']])

grouped_data['mismatch_count'] = grouped_data.apply(count_mismatches, axis=1)

# 3. Global Statistics
total_mismatches = grouped_data['mismatch_count'].sum()
total_runs = len(grouped_data) * 5
error_rate = total_mismatches / total_runs

print("-" * 40)
print(f"ALEATORIC NOISE ANALYSIS")
print("-" * 40)
print(f"Total Mismatches:      {total_mismatches} / {total_runs}")
print(f"Global Error Rate:     {error_rate:.2%}")
print("-" * 40)

# Optional: Identify the most unstable configurations
# These are the parameter sets where the simulation varies the most
grouped_data['instability_score'] = grouped_data['mismatch_count'] / 5
print("\nTop 5 Most Unstable Configurations (High Mismatch Rate):")
print(grouped_data.sort_values('mismatch_count', ascending=False).head(5)[['PT', 'y_vectors', 'instability_score']])

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 14:13:42 2026

@author: psaves
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # IMPORT THIS for custom legends
import math
import numpy as np
import os

# 1. Load Data
input_filename = 'final_aggregated_results_with_regime_v1.csv'

new_df = pd.read_csv(input_filename)

# 2. Identify variables to plot
excluded_columns = [
    'filename', 'Tick', 'Bandicoots', 'Invaders', 'Foxes',
    'Grass_Health_Index', 'Total_Bandicoots_Eaten', 'Trap_Deaths',
    'Old_Age_Deaths', 'Hunger_Deaths', 'regime', 
    'IF', 'IG', 'IR', 'IH', 'IV'
]
x_vars = [col for col in new_df.columns
          if col not in excluded_columns
          and pd.api.types.is_numeric_dtype(new_df[col])]

num_vars = len(x_vars)

if num_vars > 0:
    # Setup Grid
    ncols = 3
    nrows = math.ceil(num_vars / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows))
    axes_flat = axes.flatten() if num_vars > 1 else [axes]

    # --- EXPLICIT DEFINITIONS ---
    # 1. Define the Stack Order (Bottom to Top): Green -> Orange -> Red
    # Columns in dataframe: [2, 1, 0]
    stack_cols = [2, 1, 0]
    
    # 2. Define colors corresponding to that column order
    # Col 2 (Coex) -> Green
    # Col 1 (Prey) -> Orange
    # Col 0 (Dead) -> Red
    stack_colors = ['#1a9641', '#fdae61', '#d7191c']

    # 3. Create Custom Legend Handles (To fix the mismatch)
    # We want the legend to read Top-to-Bottom like the visual stack: Red, Orange, Green
    legend_handles = [
        mpatches.Patch(color='#d7191c', label='0: Ext'),
        mpatches.Patch(color='#fdae61', label='1: BandOnly'),
        mpatches.Patch(color='#1a9641', label='2: Coex')
    ]

    # 3. Iterate and Plot
    for i, var_name in enumerate(x_vars):
        ax = axes_flat[i]
        
        # Data Prep
        bins = pd.cut(new_df[var_name], bins=15)
        ct = pd.crosstab(bins, new_df['regime'])
        ct_norm = ct.div(ct.sum(axis=1), axis=0)
        
        # Reindex to force: [2 (Bottom), 1 (Middle), 0 (Top)]
        ct_norm = ct_norm.reindex(columns=stack_cols, fill_value=0)

        # Plot
        ct_norm.plot(kind='bar', 
                     stacked=True, 
                     ax=ax, 
                     color=stack_colors, # Colors apply to cols [2, 1, 0] in order
                     width=0.95, 
                     edgecolor='black',
                     linewidth=0.5,
                     legend=False) # Turn off auto-legend to avoid conflicts
        
        # Formatting
        ax.set_title(f'{var_name}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Proportion')
        ax.set_ylim(0, 1.0)
        
        # X-axis ticks
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        
        # Reference Line
        ax.axhline(0.5, color='white', linestyle='--', alpha=0.4)

        # Add Custom Legend ONLY to the first plot
        if i == 13:
            # bbox_to_anchor places it above the plot
            ax.legend(handles=legend_handles, 
                      loc='lower left', 
                      bbox_to_anchor=(1.02, 1.02, 1, 0.2), 
                      mode="expand", 
                      borderaxespad=0, 
                      ncol=3,
                      fontsize=9)

    # Hide unused subplots
    for j in range(num_vars, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9) # Make room for legend
    
    output_file = 'regime_proportions_corrected_legend.png'
    plt.savefig(output_file, dpi=150)
    print(f"Saved corrected plot to {output_file}")
    plt.show()
    
        
        
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math

# --- 1. Load Data ---
input_filename = "final_aggregated_results_with_regime_v1.csv"
df = pd.read_csv(input_filename)

# Define feature columns (excluding metadata and S/Seed)
excluded_columns = [
    'filename', 'Tick', 'Bandicoots', 'Invaders', 'Foxes',
    'Grass_Health_Index', 'Total_Bandicoots_Eaten', 'Trap_Deaths',
    'Old_Age_Deaths', 'Hunger_Deaths', 'regime', 
    'IF', 'IG', 'IR', 'IH', 'IV', 'S' # Ensure S is excluded
]

features = [c for c in df.columns if c not in excluded_columns and pd.api.types.is_numeric_dtype(df[c])]

print(f"Features used for grouping: {features}")

# --- 2. Group by Features to Find Replicates ---
# We group by ALL feature columns. 
# The 'regime' (y) will be aggregated to find mean and std dev.
grouped = df.groupby(features)['regime'].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()

# Filter for inputs that actually have duplicates (count > 1)
# If count == 1, we can't estimate variance from data alone.
replicates = grouped[grouped['count'] > 1].copy()

if len(replicates) == 0:
    print("No duplicated inputs found! Cannot assess robustness via replicates.")
    # Stop or handle gracefully
else:
    print(f"Found {len(replicates)} unique input configurations with replicates.")
    print(f"Average replicates per input: {replicates['count'].mean():.2f}")

    # Calculate Variance (Robustness Metric)
    replicates['variance'] = replicates['std'] ** 2
    
    # --- 3. Visualization: Replicate Variance vs Features ---
    # We plot the empirical variance observed at each design point
    
    ncols = 3
    nrows = math.ceil(len(features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    # Scaling for easier visualization (optional)
    scaler = MinMaxScaler()
    
    for i, feat in enumerate(features):
        ax = axes[i]
        
        # Scatter plot: Feature Value vs. Empirical Variance
        # We use the 'count' (number of replicates) to size the bubbles
        sc = ax.scatter(
            replicates[feat], 
            replicates['variance'], 
            alpha=0.6, 
            c=replicates['mean'], # Color by the mean regime value
            cmap='viridis',
            s=replicates['count'] * 5, # Size by number of replicates
            edgecolor='k',
            linewidth=0.5
        )
        
        ax.set_title(f"Robustness vs {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel("Empirical Variance ($s^2$)")
        ax.grid(True, alpha=0.3)
        
        # Optional: Add a trendline (lowess or rolling mean)
        # to see if variance generally increases with this feature
        # sorting for plotting line
        sorted_idx = np.argsort(replicates[feat])
        x_sorted = replicates[feat].iloc[sorted_idx]
        y_sorted = replicates['variance'].iloc[sorted_idx]
        
        # Simple rolling mean
        if len(x_sorted) > 10:
            y_roll = y_sorted.rolling(window=int(len(x_sorted)/5), min_periods=1, center=True).mean()
            ax.plot(x_sorted, y_roll, color='red', linewidth=2, label='Trend')
            ax.legend()

    # Remove empty axes
    for k in range(len(features), len(axes)):
        fig.delaxes(axes[k])

    # Colorbar for Mean Regime
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="Mean Regime Value")
    
    plt.suptitle("Empirical Robustness Assessment (Variance across Replicates)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

    # --- 4. Global Robustness Metrics ---
    mean_variance = replicates['variance'].mean()
    max_variance = replicates['variance'].max()
    
    print("\n--- Robustness Summary ---")
    print(f"Global Mean Variance (Aleatoric Uncertainty): {mean_variance:.4f}")
    print(f"Max Variance Observed: {max_variance:.4f}")
    print("Higher variance = Lower robustness to Seed (S)")
    
import pandas as pd
import scipy.stats as stats

# 1. Calculate the distribution of classes for each Seed (S)
# We count how many times 0.0, 0.5, and 1.0 appear for S=1, S=2, etc.
seed_distributions = pd.crosstab(
    index=df['S'], 
    columns=df['regime'], 
    normalize='index'  # Convert counts to percentages (0 to 1)
) * 100

print("--- Class Distribution per Seed (%) ---")
print(seed_distributions.round(2))

# 2. Check for Statistical Difference (Chi-Square Test)
# We need the raw counts (not percentages) for the Chi-Square test
contingency_table = pd.crosstab(df['S'], df['regime'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

print("\n--- Statistical Robustness (Chi-Square Test) ---")
print(f"Chi2 Statistic: {chi2:.4f}")
print(f"P-value:        {p:.4f}")

if p > 0.05:
    print("Result: No significant difference between seeds (Robust).")
else:
    print("Result: Significant difference found (Seeds affect global outcome).")
    
    
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Setup Parameters
CONFIDENCE_Z = 1.96  # For 95% Confidence
MARGIN_OF_ERROR = 0.10  # We want accuracy within +/- 0.1 (on a 0-1 scale)

# 2. Use your existing grouped data (which has 'std' calculated)
# Ensure 'grouped' comes from the previous snippet where we did .agg(['mean', 'std'])
# If not, recalculate:
feature_cols = ['PT', 'Gr', 'PM', 'PR', 'BF', 'BG', 'BR', 'BH', 'BV', 'FG', 'FR', 'FH', 'FV']
grouped = df.groupby(feature_cols)['regime'].agg(['mean', 'std']).reset_index()

# 3. Calculate Optimal N for each row
# Formula: n = (Z * sigma / E)^2
grouped['required_seeds'] = ((CONFIDENCE_Z * grouped['std']) / MARGIN_OF_ERROR) ** 2

# Round up because you can't run 3.4 simulations
grouped['required_seeds'] = np.ceil(grouped['required_seeds'])

# 4. Analysis
print("--- Optimal Seed Count Analysis ---")
print(f"Margin of Error: +/- {MARGIN_OF_ERROR}")
print(f"Confidence Level: 95%")
print("-" * 30)
print(f"Average Seeds Needed: {grouped['required_seeds'].mean():.2f}")
print(f"Max Seeds Needed:     {grouped['required_seeds'].max()}")
print(f"Percentage of points where 5 seeds were enough: {(grouped['required_seeds'] <= 5).mean():.1%}")

# 5. Visualize the "Cost" of Precision
plt.figure(figsize=(10, 5))
plt.hist(grouped['required_seeds'], bins=20, edgecolor='k', alpha=0.7)
plt.axvline(5, color='red', linestyle='--', label='Current (N=5)')
plt.xlabel('Required Seeds (n)')
plt.ylabel('Count of Configurations')
plt.title(f'Histogram of Required Seeds for E={MARGIN_OF_ERROR}')
plt.legend()
plt.show()

# 6. Show the "Worst" cases (where you need more seeds)
print("\nMost Uncertain Configurations (Need many seeds):")
print(grouped.sort_values('required_seeds', ascending=False).head(5)[['PT', 'BG', 'std', 'required_seeds']])



