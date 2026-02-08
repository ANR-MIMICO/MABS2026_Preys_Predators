import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from matplotlib import rcParams
from smt.sampling_methods import LHS

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

# 1. Load Data
input_filename = 'final_aggregated_results_with_regime.csv'
df = pd.read_csv(input_filename)

# Fix column name first
if "PT" in df.columns:
    df = df.rename(columns={"PT": "PH"})

features = ['PH', 'Gr', 'PR', 'PM', 'BF', 'BG', 'BR', 'BH', 'BV', 'FR','FG', 'FH', 'FV']
X = df[features]
y = df['regime']

# 2. Train Model
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X)

# Classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 128), 
    activation='relu',            
    solver='adam',                
    alpha=0.01,                   
    max_iter=5000,                
    random_state=42
)
mlp.fit(X_train_scaled, y)

# 3. LHS Sampling (in Normalized Space [0,1])
n_features = len(features)
# Bounds for LHS in normalized space are just 0 to 1 for every feature
xlimits = np.array([[0.0, 1.0]] * n_features)

sampling = LHS(xlimits=xlimits, criterion='ese', random_state=42)
n_samples = 300 
X_lhs_scaled = sampling(n_samples)

X_lhs_df = pd.DataFrame(X_lhs_scaled, columns=features)

color_feature = 'BG' 
color_idx = features.index(color_feature)
color_values_norm = X_lhs_scaled[:, color_idx] # These are already 0-1

# Setup Colormap
cmap = plt.cm.jet
norm = plt.Normalize(vmin=0, vmax=1) 

# Prepare Grid
ncols = 3
nrows = math.ceil(n_features / ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 5 * nrows))
axes_flat = axes.flatten()

print(f"Generating PDP + ICE (Class 2 Coexistence)...")

# 5. The Loop
for i, feat_name in enumerate(features):
    ax = axes_flat[i]
    
    # Calculate Partial Dependence
    # We use 'predict_proba' to get the probability score
    pd_results = partial_dependence(
        mlp, X_lhs_df, [feat_name], kind='both', 
        grid_resolution=200,
       response_method= "predict_proba"
    )
    
    # Extract results for Class 2 (Coexistence)
    # pd_results['average'] shape: (n_classes, n_grid_points)
    pdp_y = pd_results['average'][2] 
    
    # pd_results['individual'] shape: (n_classes, n_samples, n_grid_points)
    ice_y =pd_results['individual'][2]

    # Extract Grid (X-axis is scaled [0,1])
    grid_scaled = pd_results['grid_values'][0]
    
    # --- UNSCALE X-AXIS ONLY ---
    feat_idx = features.index(feat_name)
    min_x = scaler.data_min_[feat_idx]
    max_x = scaler.data_max_[feat_idx]
    grid_orig = grid_scaled * (max_x - min_x) + min_x
    
    # --- Y-AXIS: RAW COMPUTED VALUES ---
    # We do NOT unscale Y. We do NOT force limits. 
    # We assume 'predict_proba' returns values (0 to 1) and we plot them as is.
    
    # Smoothing ICE curves for better visualization
    ice_smooth = gaussian_filter1d(ice_y, sigma=2, axis=1)
    
    # Plot ICE Curves
    for j in range(ice_smooth.shape[0]):
        c_val = color_values_norm[j] 
        ax.plot(grid_orig, ice_smooth[j], color=cmap(norm(c_val)), alpha=0.20, linewidth=0.75)
        
    # Plot PDP Curve
    ax.plot(grid_orig, pdp_y, color='black', linewidth=3, label='PDP')
    
    # Formatting
    ax.set_title(f"{feat_name}")
    ax.set_xlabel(feat_name)
    ax.set_ylabel("Partial Dependence") # Restored Label
    
    # REMOVED: ax.set_ylim(-0.05, 1.05) 
    # Let matplotlib auto-scale the Y-axis to show the actual range of variation
    
    ax.grid(True, linestyle='--', alpha=0.3)

# 6. Final Layout & Colorbar
# Hide unused subplots
for k in range(n_features, len(axes_flat)):
    fig.delaxes(axes_flat[k])

plt.tight_layout(rect=[0, 0, 0.9, 1]) 
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)

# Fix Colorbar Labels
c_min = scaler.data_min_[color_idx]
c_max = scaler.data_max_[color_idx]
ticks = np.linspace(0, 1, 5)
tick_labels = [f"{t * (c_max - c_min) + c_min:.0f}" for t in ticks] # Removed % to be generic
cbar.set_ticks(ticks)
cbar.set_ticklabels(tick_labels)
cbar.set_label(f"{color_feature} Value", fontsize=14)

output_file = 'pdp_ice_mlp_raw_axis.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Plot saved to {output_file}")
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from smt.sampling_methods import LHS
from scipy.ndimage import gaussian_filter1d
import math
import os

# Configuration
plt.rcParams.update({'font.size': 12, 'figure.titlesize': 16})

# ============================================================
# 1. LOAD DATA
# ============================================================
input_filename = "final_aggregated_results_with_regime.csv"
if not os.path.exists(input_filename):
    print("Error: File not found. Creating dummy data...")
    df = pd.DataFrame(np.random.rand(500, 14), columns=['PH', 'Gr', 'PR', 'PM', 'BF', 'BG', 'BR', 'BH', 'BV', 'FR','FG', 'FH', 'FV', 'regime'])
    df['regime'] = np.random.randint(0, 3, 500)
else:
    df = pd.read_csv(input_filename)

if "PT" in df.columns:
    df = df.rename(columns={"PT": "PH"})

features = ['PH', 'Gr', 'PR', 'PM', 'BF', 'BG', 'BR', 'BH', 'BV', 'FR','FG', 'FH', 'FV']
X = df[features].values
y = df["regime"].values

# Scale
scaler_x = MinMaxScaler()
X_scaled = scaler_x.fit_transform(X)

# ============================================================
# 2. MODEL A: CLASSIFIER (Epistemic Source)
# ============================================================
print("1/3 Training Classifier (Mean Prediction)...")

rf_clf = RandomForestClassifier(
    n_estimators=300, 
    min_samples_leaf=5,
    n_jobs=-1,
    oob_score=True, # Critical for calculating residuals
    random_state=42
)
rf_clf.fit(X_scaled, y)

# --- Identify Class 2 Index ---
target_class = 2
try:
    class_idx = np.where(rf_clf.classes_ == target_class)[0][0]
    print(f"Target Class '{target_class}' is at index {class_idx}")
except IndexError:
    raise ValueError(f"Class {target_class} not found!")

# ============================================================
# 3. CALCULATE RESIDUALS FOR PROBABILITY
# ============================================================
# 1. Get the OOB Probabilities (The "Honest" predictions on training data)
# Shape: (n_samples, n_classes)
oob_probs = rf_clf.oob_decision_function_ 
prob_class_2 = oob_probs[:, class_idx]

# 2. Create Binary Truth for Class 2 (1 if Coexistence, 0 if not)
y_binary = (y == target_class).astype(int)

# 3. Calculate Aleatoric Residuals (How wrong was the probability?)
# If y=1 and Prob=0.8, Error=0.2. If y=1 and Prob=0.4, Error=0.6 (High Noise)
abs_residuals = np.abs(y_binary - prob_class_2)

# ============================================================
# 4. MODEL B: REGRESSOR (Aleatoric Source)
# ============================================================
print("2/3 Training Sigma Regressor (Aleatoric Source)...")

# We use a Regressor to predict the *magnitude* of the probability error
rf_sigma = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=5, 
    n_jobs=-1,
    random_state=42
)
rf_sigma.fit(X_scaled, abs_residuals)

# ============================================================
# 5. EUCLIDEAN SUM FUNCTION (PROBABILITY VERSION)
# ============================================================
def get_total_uncertainty_proba(model_clf, model_sigma, X_batch, target_idx):
    """
    Returns Total Uncertainty = sqrt( Aleatoric^2 + Epistemic^2 )
    Applied to the Probability of Class 2.
    """
    # A. Aleatoric: Predicted Residual (System Noise)
    # "How far is the probability usually from the truth?"
    sigma_aleatoric = model_sigma.predict(X_batch)
    
    # B. Epistemic: Std Dev of Trees (Model Disagreement)
    # "How much do the trees disagree on the probability?"
    all_tree_probs = np.stack([tree.predict_proba(X_batch)[:, target_idx] for tree in model_clf.estimators_])
    sigma_epistemic = np.std(all_tree_probs, axis=0)
    
    # C. Euclidean Sum
    sigma_total = np.sqrt(sigma_aleatoric**2 + sigma_epistemic**2)
    
    return sigma_total

# ============================================================
# 6. PLOTTING LOOP
# ============================================================
print("3/3 Generating Euclidean Probability Uncertainty Plots...")

n_features = len(features)
lhs = LHS(xlimits=np.array([[0.0, 1.0]] * n_features), random_state=42)
X_lhs = lhs(300) 

color_feature = 'BG'
if color_feature not in features: color_feature = features[0]
color_idx = features.index(color_feature)
color_values = X_lhs[:, color_idx]

# Setup
ncols = 3
nrows = math.ceil(n_features / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
axes = axes.flatten()

cmap = plt.cm.jet
norm = plt.Normalize(vmin=0, vmax=1)

grid_res = 100
grid = np.linspace(0, 1, grid_res)

for i, feat_name in enumerate(features):
    ax = axes[i]
    feat_idx = features.index(feat_name)
    
    # Batch Prep
    X_batch = []
    for sample in X_lhs:
        mat = np.tile(sample, (grid_res, 1))
        mat[:, feat_idx] = grid
        X_batch.append(mat)
    X_batch = np.vstack(X_batch)
    
    # --- GET TOTAL UNCERTAINTY ---
    sigma_total_pred = get_total_uncertainty_proba(rf_clf, rf_sigma, X_batch, class_idx)
    
    # Reshape
    ice_curves = sigma_total_pred.reshape(len(X_lhs), grid_res)
    ice_curves = gaussian_filter1d(ice_curves, sigma=2, axis=1)
    pdp = ice_curves.mean(axis=0)
    
    # X-Axis
    xmin = scaler_x.data_min_[feat_idx]
    xmax = scaler_x.data_max_[feat_idx]
    x_orig = grid * (xmax - xmin) + xmin

    # Plot
    for sample_idx in range(len(ice_curves)):
        c_val = color_values[sample_idx]
        ax.plot(x_orig, ice_curves[sample_idx], color=cmap(norm(c_val)), alpha=0.15, linewidth=0.75)
        
    ax.plot(x_orig, pdp, color='black', lw=2.5, label='Mean Total Uncertainty')
    
    ax.set_title(feat_name)
    if i % ncols == 0:
        ax.set_ylabel(r"$\sigma_{total}$ on $P(C)$")
        
    ax.grid(True, linestyle='--', alpha=0.3)

# Cleanup
for k in range(n_features, len(axes)):
    fig.delaxes(axes[k])

plt.tight_layout(rect=[0, 0, 0.9, 1])
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)

c_min = scaler_x.data_min_[color_idx]
c_max = scaler_x.data_max_[color_idx]
ticks = np.linspace(0, 1, 5)
labels = [f"{t*(c_max-c_min)+c_min:.1f}" for t in ticks]
cbar.set_ticks(ticks)
cbar.set_ticklabels(labels)
cbar.set_label(f"Colored by {color_feature}")

plt.suptitle(f"Total Uncertainty on Coexistence Probability\n(Euclidean Sum of Aleatoric + Epistemic)", fontsize=16)
plt.savefig("total_uncertainty_proba_euclidean.png", dpi=150, bbox_inches='tight')
print("Done.")
plt.show()







