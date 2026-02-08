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



df_stats = pd.DataFrame(X_scaled, columns=features)
df_stats['Target_Regime'] = y_scaled 

# 2. Création de la formule (Target ~ Var1 + Var2 + ...)
# On écrit une équation linéaire
formula_string = 'Target_Regime ~ ' + ' + '.join(features)
print(f"Modèle : {formula_string}")

# 3. Ajustement du Modèle Linéaire (OLS - Ordinary Least Squares)
# C'est la base des GLM gaussiens.
model = ols(formula_string, data=df_stats).fit()

# 4. Calcul de l'ANOVA (Analyse de la Variance) de Type II
# Le Type II est important si vos variables ne sont pas parfaitement orthogonales
anova_table = sm.stats.anova_lm(model, typ=2)

# 5. Calcul du pourcentage de variance expliquée (Eta-Squared)
# La somme des carrés (sum_sq) totale inclut le "Residual" (ce qui n'est pas expliqué)
total_sum_sq = anova_table['sum_sq'].sum()
anova_table['% Variance Expliquée (eta_sq)'] = (anova_table['sum_sq'] / total_sum_sq) * 100

# 6. Affichage propre
# On trie par importance et on enlève la ligne "Residual" pour voir les gagnants
results = anova_table.sort_values(by='% Variance Expliquée (eta_sq)', ascending=False)

print("\n--- Décomposition de la Variance (Approche Linéaire) ---")
print(results[['sum_sq', 'F', 'PR(>F)', '% Variance Expliquée (eta_sq)']])

# Petit check global
r_squared = model.rsquared * 100
print(f"\nLe modèle linéaire donne R2={r_squared:.2f}%.")


# 1. Prepare Data
# We focus on the top drivers found by ANOVA to make the tree cleaner, 
# but you can use 'features' (all variables) if you prefer.
X_tree =  df[features].values
y_tree = y_scaled
regime_names = ['ALL DEAD', 'FOXES DEAD', 'ALL ALIVE'] # Adjust based on your data
# 2. Train the Tree
# max_depth=3 ensures we only get the most critical thresholds (the "big picture")
tree_reg = DecisionTreeRegressor(max_leaf_nodes=6, max_depth=6, random_state=42)
tree_reg.fit(X_tree, y_tree)
# 3. Print the Thresholds as Text (Readability)
print("\n--- Key Thresholds Identified (Decision Tree Rules) ---")
tree_rules = export_text(tree_reg, feature_names=features)
print(tree_rules)

# 4. Visualize the Tree
plt.figure(figsize=(20, 10))
plot_tree(tree_reg, 
          feature_names=features, 
          class_names=regime_names,  # <--- ADD THIS
          filled=True, 
          rounded=True, 
          fontsize=12,
          precision=2)
plt.title("Decision Tree: Tipping Points for Regime Change")

plt.savefig("dt.png", dpi=300, bbox_inches='tight') 
print("Plot saved to dt.png")

plt.show()


