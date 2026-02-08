import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Added for Heatmap
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams
from SALib.sample import saltelli
from SALib.analyze import sobol

# Configuration for nice plots
rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

input_filename = 'final_aggregated_results_with_regime.csv'
df = pd.read_csv(input_filename)
features = ['PT', 'Gr', 'PR', 'PM', 'BF', 'BG', 'BR', 'BH', 'BV', 'FR','FG', 'FH', 'FV']
X = df[features]
y = df['regime']

#y = df['Bandicoots']+df['Bandicoots']*df['Foxes']

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X)
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 128), 
    activation='relu',            
    solver='adam',                
    alpha=0.01,                   
    max_iter=5000,                
    random_state=42
)
mlp.fit(X_train_scaled, y)


problem = {
    'num_vars': len(features),
    'names': features,
    'bounds': [[0, 1]] * len(features) # Scaled bounds [0, 1]
}
N =32768
param_values = saltelli.sample(problem, N, calc_second_order=True)
#Y_surrogate = mlp.predict(param_values)
Y_surrogate = mlp.predict_proba(param_values)[:, 2]

Si = sobol.analyze(problem, Y_surrogate, calc_second_order=True, print_to_console=False)
df_sobol = pd.DataFrame({
    "Parameter": features,
    "S1": Si['S1'],
    "ST": Si['ST'],
    "S1_conf": Si['S1_conf'],
    "ST_conf": Si['ST_conf']
})


# 3. SORT BY S1 (Descending) <--- THIS IS THE CHANGE
df_sobol = df_sobol.sort_values(by='S1', ascending=False).reset_index(drop=True)

df_sobol.iloc[3, 0] = 'PH'
print("\nTop Sensitivity Drivers (Sorted by S1):")
print(df_sobol.head(5))

# --- PLOT 1: Bar Chart (Sorted) ---
fig_mlp, ax_mlp = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(df_sobol))  # Use len of sorted DF
width = 0.35

# Plot bars using the sorted DataFrame
ax_mlp.bar(x_pos - width/2, df_sobol['S1'], width, yerr=df_sobol['S1_conf'], 
           label='S1 (Main Effect)', color='#ff9999', capsize=5)
ax_mlp.bar(x_pos + width/2, df_sobol['ST'], width, yerr=df_sobol['ST_conf'], 
           label='ST (Total Effect)', color='#cc0000', capsize=5)

ax_mlp.set_xticks(x_pos)
# Use the sorted parameter names
ax_mlp.set_xticklabels(df_sobol['Parameter'], rotation=45, ha='right')

ax_mlp.set_ylabel('Sobol Index (Variance Contribution)')
ax_mlp.set_title('Global Sensitivity Analysis (Sorted by First Order Effect)')
ax_mlp.legend()
ax_mlp.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('sobol_indices_mlp_sorted.png', dpi=300)
print("Plot saved: sobol_indices_mlp_sorted.png")
plt.show()


features2 = ['PH', 'Gr', 'PR', 'PM', 'BF', 'BG', 'BR', 'BH', 'BV', 'FR','FG', 'FH', 'FV']
sorted_features = df_sobol['Parameter'].tolist()
interaction_matrix = np.array(Si['S2'])
interaction_matrix = np.nan_to_num(interaction_matrix) 
interaction_matrix = np.clip(interaction_matrix, a_min=0, a_max=None) * 100

# 2. THE FIX: Make it Symmetric (Mirror Upper to Lower)
# We add the Transpose to fill the bottom half
interaction_matrix = interaction_matrix + interaction_matrix.T

# 3. Create DataFrame
df_interactions = pd.DataFrame(interaction_matrix, index=features2, columns=features2)

# 4. Sort Rows and Columns by S1 Importance
# (Using the sorted list 'sorted_features' from the previous step)
# sorted_features = df_sobol.sort_values(by='ST', ascending=False)['Parameter'].tolist()
df_interactions_sorted = df_interactions.loc[sorted_features, sorted_features]

# 5. Plot
plt.figure(figsize=(12, 10))

# OPTIONAL: We still mask the diagonal (Self-interaction is always 0)
mask = np.eye(len(df_interactions_sorted), dtype=bool)
mask = np.tril(np.ones_like(df_interactions_sorted, dtype=bool))
sns.heatmap(df_interactions_sorted, 
            annot=True, 
            fmt=".1f", 
            cmap="viridis", 
            mask=mask,   # Only mask the diagonal line
            cbar_kws={'label': 'Second Order Sobol Index (S2) %'})

plt.title("Parameter Interactions (Sorted by Total Effect)")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('sobol_interactions_heatmap_sorted_full.png', dpi=300)
print("Plot saved.")
plt.show()