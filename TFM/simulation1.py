# -*- coding: utf-8 -*-
"""
Simulation du dilemme du prisonnier avec vigilance (L=1)
Courbes ⟨ρ⟩ en fonction de b pour plusieurs θ, moyennées sur 25 runs
"""

import numpy as np
import matplotlib.pyplot as plt

from tfm_pg1 import run_simulation  # ⚠️ Vérifie le nom exact du module

# ----------------------- PARAMÈTRES GLOBAUX -----------------------
N = 200                 # Nombre de nœuds
network_type = 'er'     # 'er' ou 'ba'
z = 4                   # Degré moyen
L = 4                   # Portée des interactions (L=1 = voisins directs)
lam = 0.5               # Facteur d’atténuation
R, S, P = 1.0, 0.0, 0.0 # Paiements
generations = 200       # Durée d'une simulation
initial_coop_frac = 0.5 # Fraction initiale de coopérateurs

# ------------------ PARAMÈTRES DE BALAYAGE ------------------
b_values = np.linspace(1.0, 2.0, 11)      # Valeurs du paramètre b
theta_values = [0, 0.2, 0.4, 0.6, 0.8, 1] # Seuils de vigilance
repetitions = 25                          # Moyenne sur 25 runs
base_seed = 1000                          # Première graine aléatoire
# ------------------------------------------------------------

# Tableaux de stockage
rho_means = np.zeros((len(theta_values), len(b_values)))

for t_idx, theta_val in enumerate(theta_values):
    for b_idx, b_val in enumerate(b_values):
        coop_vals = []
        for rep in range(repetitions):
            seed = base_seed + rep
            res = run_simulation(
                N=N, network_type=network_type, z=z, L=L, lam=lam,
                R=R, T_param=b_val, S=S, P=P,
                generations=generations, seed=seed,
                theta_global=theta_val, initial_coop_frac=initial_coop_frac
            )
            coop_vals.append(res["final_coop"])
        
        rho_means[t_idx, b_idx] = np.mean(coop_vals)
        print(f"θ={theta_val:.1f}, b={b_val:.2f} → ⟨ρ⟩={rho_means[t_idx, b_idx]:.3f}")

# ------------------ GRAPHIQUE ⟨ρ⟩ vs b ------------------
plt.figure(figsize=(8,5))
for t_idx, theta_val in enumerate(theta_values):
    plt.plot(b_values, rho_means[t_idx], marker='o', label=f"θ = {theta_val}")

plt.xlabel("b (T_param)")
plt.ylabel("⟨ρ⟩ (fraction moyenne de coopérateurs)")
plt.title(f"Évolution de ⟨ρ⟩ en fonction de b (L={L}, z={z}, moy. sur 25 runs)")
plt.ylim(-0.05, 1.05)
plt.legend(title="θ")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"rho_vs_b_L{L}_mean25.png", dpi=150)
plt.show()

# ------------------ SAUVEGARDE CSV ------------------
header = "b," + ",".join([f"theta_{th}" for th in theta_values])
data_to_save = np.column_stack([b_values, rho_means.T])
np.savetxt(f"rho_vs_b_L{L}{network_type}_mean25.csv", data_to_save, delimiter=",", header=header, comments="")
print(f"> Résultats sauvegardés dans rho_vs_b_L{L}_{network_type}_Z{z}_mean25.csv et rho_vs_b_L{L}{network_type}_mean25.png")
