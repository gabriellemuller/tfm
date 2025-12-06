import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

# Importation du modèle AVEC COÛT
# Assurez-vous que tfm_pg1_confermi_cost.py est dans le même dossier
try:
    from tfm_pg1_confermi_cost import run_simulation
except ImportError:
    # Fallback au cas où le nom serait différent
    from tfm_pg1_confermi import run_simulation

# =============================================================================
# 1. PARAMÈTRES DE L'ÉTUDE
# =============================================================================
N = 200
NETWORK_TYPE = 'er'
Z = 4                  # Réseau dense
LAMBDA = 0.5
K_FERMI = 0.1

# Paramètres variables
L_VALUES = [1, 4]       # On compare Local vs Longue Portée
COST_VALUES = [0.25, 0.50, 0.75] # 25%, 50%, 75% de R
THETA_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
B_VALUES = np.linspace(1.0, 2.0, 11)

REPETITIONS = 25        # Robustesse statistique

# =============================================================================
# 2. FONCTION WRAPPER (Pour la parallélisation)
# =============================================================================
def single_run(rep, b, theta, cost, L_val):
    # Graine unique
    my_seed = 1000 + int(b*100) + int(theta*100) + rep + int(cost*100) + L_val
    
    res = run_simulation(
        N=N, network_type=NETWORK_TYPE, z=Z, L=L_val, lam=LAMBDA,
        R=1.0, T_param=b, S=0.0, P=0.0,
        generations=200, seed=my_seed,
        theta_global=theta, initial_coop_frac=0.5,
        K_FERMI=K_FERMI, COST_PARAM=cost
    )
    return res['final_coop'], res['final_vig']

# =============================================================================
# 3. MOTEUR PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    start_time_global = time.time()
    print(f"--- DÉBUT DE L'ÉTUDE COMPARATIVE L1 vs L4 (ER Z={Z}) ---")
    
    # Dictionnaire pour stocker les résultats en mémoire pour les plots
    # Structure: all_results[cost][L][theta] = [rho_mean_array, vig_mean_array]
    all_results = {}

    # --- BOUCLE SUR LES COÛTS ---
    for cost in COST_VALUES:
        print(f"\n>>> TRAITEMENT COÛT = {cost*100}% ...")
        all_results[cost] = {}
        
        # --- BOUCLE SUR LA PORTÉE (L=1, L=4) ---
        for L_val in L_VALUES:
            print(f"   > Configuration L={L_val}...")
            
            # Matrices pour stocker les résultats de ce L et ce Coût
            # Lignes = Theta, Colonnes = b
            rho_matrix = np.zeros((len(THETA_VALUES), len(B_VALUES)))
            vig_matrix = np.zeros((len(THETA_VALUES), len(B_VALUES)))
            
            all_results[cost][L_val] = {}

            # --- BOUCLE SUR THETA ---
            for t_idx, theta in enumerate(THETA_VALUES):
                # Parallélisation sur b pour aller plus vite
                # On lance tous les b et toutes les répétitions pour ce theta
                
                # Petite astuce : on aplatit la boucle b et rep pour paralléliser au max
                tasks = []
                for b in B_VALUES:
                    for r in range(REPETITIONS):
                        tasks.append((b, r))
                
                results = Parallel(n_jobs=-1)(
                    delayed(single_run)(r, b, theta, cost, L_val) for b, r in tasks
                )
                
                # Reconstitution des moyennes
                # results est une liste de taille len(B_VALUES) * REPETITIONS
                # On doit regrouper par b
                for b_idx, b in enumerate(B_VALUES):
                    # Les indices correspondant à ce b dans la liste plate 'results'
                    start_i = b_idx * REPETITIONS
                    end_i = (b_idx + 1) * REPETITIONS
                    subset = results[start_i:end_i]
                    
                    coops = [x[0] for x in subset]
                    vigs = [x[1] for x in subset]
                    
                    rho_matrix[t_idx, b_idx] = np.mean(coops)
                    vig_matrix[t_idx, b_idx] = np.mean(vigs)
            
            # Stockage mémoire pour le plot
            all_results[cost][L_val]['rho'] = rho_matrix
            all_results[cost][L_val]['vig'] = vig_matrix

            # --- SAUVEGARDE CSV IMMÉDIATE ---
            # 1. Coopération
            df_rho = pd.DataFrame(rho_matrix.T, columns=[f"theta_{t}" for t in THETA_VALUES])
            df_rho.insert(0, "b", B_VALUES)
            csv_rho = f"rho_L{L_val}_cost{cost}_ER_Z{Z}.csv"
            df_rho.to_csv(csv_rho, index=False, sep=',', decimal='.')
            
            # 2. Vigilance
            df_vig = pd.DataFrame(vig_matrix.T, columns=[f"theta_{t}" for t in THETA_VALUES])
            df_vig.insert(0, "b", B_VALUES)
            csv_vig = f"vig_L{L_val}_cost{cost}_ER_Z{Z}.csv"
            df_vig.to_csv(csv_vig, index=False, sep=',', decimal='.')
            
            print(f"     Données sauvegardées : {csv_rho} et {csv_vig}")

    # =============================================================================
    # 4. GÉNÉRATION DES GRAPHIQUES COMPARATIFS
    # =============================================================================
    print("\n--- GÉNÉRATION DES GRAPHIQUES ---")
    
    # On génère 1 figure par Coût
    for cost in COST_VALUES:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Sous-graphe 1 : Coopération (L1 vs L4)
        ax_coop = axes[0]
        # On ne trace que quelques thetas clés pour lisibilité (ex: 0.0, 0.4, 0.8)
        thetas_to_plot = [0.0, 0.4, 0.8]
        colors = ['blue', 'green', 'red']
        
        for i, theta in enumerate(thetas_to_plot):
            if theta in THETA_VALUES:
                idx = THETA_VALUES.index(theta)
                # L=1 (Ligne pleine)
                ax_coop.plot(B_VALUES, all_results[cost][1]['rho'][idx], 
                             color=colors[i], linestyle='-', marker='o', 
                             label=f"L=1 $\\theta={theta}$")
                # L=4 (Pointillés)
                ax_coop.plot(B_VALUES, all_results[cost][4]['rho'][idx], 
                             color=colors[i], linestyle='--', marker='x', 
                             label=f"L=4 $\\theta={theta}$")
        
        ax_coop.set_title(f"Coopération (Coût={int(cost*100)}%)")
        ax_coop.set_xlabel("Tentation ($b$)")
        ax_coop.set_ylabel("Fraction $\\rho$")
        ax_coop.set_ylim(-0.05, 1.05)
        ax_coop.grid(True, alpha=0.3)
        ax_coop.legend()

        # Sous-graphe 2 : Vigilance (L1 vs L4)
        ax_vig = axes[1]
        for i, theta in enumerate(thetas_to_plot):
            if theta in THETA_VALUES:
                idx = THETA_VALUES.index(theta)
                # L=1 (Ligne pleine)
                ax_vig.plot(B_VALUES, all_results[cost][1]['vig'][idx], 
                            color=colors[i], linestyle='-', marker='o')
                # L=4 (Pointillés)
                ax_vig.plot(B_VALUES, all_results[cost][4]['vig'][idx], 
                            color=colors[i], linestyle='--', marker='x')
        
        ax_vig.set_title(f"Vigilance (Coût={int(cost*100)}%)")
        ax_vig.set_xlabel("Tentation ($b$)")
        ax_vig.set_ylabel("Fraction $V$")
        ax_vig.set_ylim(-0.05, 1.05)
        ax_vig.grid(True, alpha=0.3)
        # Pas de légende ici pour ne pas surcharger (mêmes codes couleurs)

        plt.suptitle(f"Comparaison Local (L1) vs Longue Portée (L4) - ER Z={Z}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"comparaison_L1_L4_cost{cost}_ER_Z{Z}.png")
        print(f"Graphique sauvegardé : comparaison_L1_L4_cost{cost}_ER_Z{Z}.png")

    total_time = time.time() - start_time_global
    print(f"\n--- ÉTUDE TERMINÉE EN {total_time/60:.1f} MINUTES ---")