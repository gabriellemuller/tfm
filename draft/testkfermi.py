# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 19:06:39 2025

@author: gabri
"""

# -*- coding: utf-8 -*-
"""
Script de Calibration du paramètre de bruit K (Fermi)
Objectif : Justifier le choix de K=0.1 pour le TFM.
"""

import numpy as np
import matplotlib.pyplot as plt
from tfm_pg1_confermi import run_simulation  # Assure-toi que ce fichier est dans le même dossier

# ----------------------- PARAMÈTRES FIXES -----------------------
N = 200                 # Population
network_type = 'ba'     # Barabási-Albert
z = 16                  # Degré moyen (Densité moyenne)
L = 4                   # Modèle étendu
lam = 0.5               # Atténuation
theta_fixe = 0.4        # Seuil de vigilance moyen
b_fixe = 1.2            # Tentation modérée (challenge pour la coopération)

generations = 200       # Durée pour atteindre l'état stationnaire
repetitions = 20        # Nombre de répétitions pour lisser la courbe
base_seed = 999         # Pour la reproductibilité

# ------------------ PARAMÈTRES VARIABLES (K) ------------------
# Échelle logarithmique pour bien voir les ordres de grandeur
k_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

# Stockage des résultats
rho_means = []
rho_stds = []

print(f"--- DÉBUT DE LA CALIBRATION DE K ---")
print(f"Config: N={N}, b={b_fixe}, L={L}, z={z}, θ={theta_fixe}")

for k_val in k_values:
    coop_vals = []
    
    for rep in range(repetitions):
        # On change la seed à chaque rep pour avoir des réseaux différents
        current_seed = base_seed + rep + (int(k_val*1000)) 
        
        res = run_simulation(
            N=N, 
            network_type=network_type, 
            z=z, 
            L=L, 
            lam=lam,
            R=1.0, 
            T_param=b_fixe,  # On fixe la tentation
            S=0.0, 
            P=0.0,
            generations=generations, 
            seed=current_seed,
            theta_global=theta_fixe, 
            initial_coop_frac=0.5,
            K_FERMI=k_val    # <--- C'est ici qu'on fait varier K
        )
        coop_vals.append(res["final_coop"])
    
    mean_rho = np.mean(coop_vals)
    std_rho = np.std(coop_vals)
    rho_means.append(mean_rho)
    rho_stds.append(std_rho)
    
    print(f"K = {k_val:6.3f} -> <ρ> = {mean_rho:.3f} (±{std_rho:.3f})")

# ------------------ GRAPHIQUE ------------------
plt.figure(figsize=(8, 6))

# On utilise une échelle logarithmique pour l'axe X (car K varie de 0.001 à 5)
plt.semilogx(k_values, rho_means, marker='o', linestyle='-', color='b', label='Moyenne Coopération')

# Zone d'écart-type (optionnel, pour faire pro)
plt.fill_between(k_values, 
                 np.array(rho_means) - np.array(rho_stds), 
                 np.array(rho_means) + np.array(rho_stds), 
                 color='b', alpha=0.2, label='Écart-type')

plt.xlabel("Paramètre de Bruit K (Échelle Log)")
plt.ylabel("Fraction Finale de Coopérateurs ⟨ρ⟩")
plt.title(f"Calibration de K (b={b_fixe}, z={z}, L={L})")
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.legend()

# Ligne verticale pour montrer ton choix
plt.axvline(x=0.1, color='r', linestyle='--', label='Choix K=0.1')
plt.legend()

filename = "calibration_K_logscale.png"
plt.savefig(filename, dpi=150)
plt.show()

print(f"> Graphique sauvegardé sous {filename}")