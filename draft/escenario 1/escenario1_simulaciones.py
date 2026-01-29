import numpy as np
import networkx as nx
import pandas as pd
import os
from joblib import Parallel, delayed
from tqdm import tqdm

# ==========================================
# ⚙️ CONFIGURATION (LOCALE)
# ==========================================
N_AGENTS = 1000         
GENERATIONS = 300       
REPETITIONS = 15        # 15 essais pour avoir la variance
K_FERMI = 0.1           # <--- C'EST LA BONNE VALEUR (Sélection forte)
LAMBDA_DECAY = 0.5      
SAVE_PATH = 'Resultats_Local'  # Dossier qui sera créé à côté du script

# Création du dossier si inexistant
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

print(f"🔬 INITIALISATION LOCAL | K={K_FERMI} | N={N_AGENTS}")

# ==========================================
# 🛠️ FONCTIONS
# ==========================================

def precompute_layers(G, L, lam):
    nodes = list(G.nodes())
    layer_data = {}
    raw_weights = [lam**(l-1) for l in range(1, L+1)]
    for i in nodes:
        lengths = nx.single_source_shortest_path_length(G, i, cutoff=L)
        layers = {l: [] for l in range(1, L+1)}
        for target, dist in lengths.items():
            if dist > 0: layers[dist].append(target)
        present_weights = []
        valid_layers = []
        for l in range(1, L+1):
            if len(layers[l]) > 0:
                present_weights.append(raw_weights[l-1])
                valid_layers.append(l)
        if sum(present_weights) > 0:
            norm_weights = [w/sum(present_weights) for w in present_weights]
        else:
            norm_weights = []
        layer_data[i] = {'layers': layers, 'valid_indices': valid_layers, 'alphas': norm_weights, 'k_l': {l: len(layers[l]) for l in range(1, L+1)}}
    return layer_data

def run_single_replica(rep_id, N, z, L, b, theta):
    # Création du réseau
    m = int(z / 2)
    G = nx.barabasi_albert_graph(N, m)
    
    layers_data = precompute_layers(G, L, LAMBDA_DECAY)
    strategies = np.random.choice([0, 1], size=N)
    vigilance = strategies.copy()
            
    for t in range(GENERATIONS):
        # 1. Influence & Tentation
        I = np.zeros(N)
        T_eff = np.zeros(N)
        for i in range(N):
            data = layers_data[i]
            i_val = 0
            for idx, l in enumerate(data['valid_indices']):
                m_l = sum(vigilance[n] for n in data['layers'][l])
                i_val += data['alphas'][idx] * (m_l / data['k_l'][l])
            I[i] = i_val
            T_eff[i] = 1 + (b - 1) * (1 - I[i])
        
        # 2. Update Vigilance
        new_vigilance = np.zeros(N)
        mask_active = (strategies == 1) & (I >= theta)
        new_vigilance[mask_active] = 1
        vigilance = new_vigilance
        
        # 3. Payoffs
        payoffs = np.zeros(N)
        for u, v in G.edges():
            # u vs v
            if strategies[u] == 1: payoffs[u] += 1 if strategies[v] == 1 else 0
            else: payoffs[u] += T_eff[u] if strategies[v] == 1 else 0
            # v vs u
            if strategies[v] == 1: payoffs[v] += 1 if strategies[u] == 1 else 0
            else: payoffs[v] += T_eff[v] if strategies[u] == 1 else 0
            
        # 4. Fermi Rule
        new_strategies = strategies.copy()
        for i in range(N):
            nbs = list(G.neighbors(i))
            if len(nbs) > 0:
                j = np.random.choice(nbs)
                delta_pi = payoffs[j] - payoffs[i]
                prob = 1 / (1 + np.exp(-delta_pi / K_FERMI))
                if np.random.random() < prob:
                    new_strategies[i] = strategies[j]
        strategies = new_strategies
        
    return np.mean(strategies)

def run_scenario_parallel(name, z, L, b_list, theta_list):
    print(f"\n--- ⚡ LANCEMENT {name} (K={K_FERMI}) ---")
    tasks = []
    for theta in theta_list:
        for b in b_list:
            for rep in range(REPETITIONS):
                tasks.append((rep, N_AGENTS, z, L, b, theta))
    
    # Exécution locale sur tous tes cœurs CPU
    results = Parallel(n_jobs=-1, verbose=5)(delayed(run_single_replica)(*t) for t in tqdm(tasks))
    
    df = pd.DataFrame(tasks, columns=['rep', 'N', 'z', 'L', 'b', 'theta'])
    df['rho'] = results
    
    # Calcul Moyenne + Ecart-type
    df_final = df.groupby(['b', 'theta'])['rho'].agg(['mean', 'std']).reset_index()
    df_final.columns = ['b', 'theta', 'rho_mean', 'rho_std']
    
    # Sauvegarde locale
    filename = os.path.join(SAVE_PATH, f"{name}.csv")
    df_final.to_csv(filename, index=False)
    print(f"✅ Sauvegardé : {filename}")

# ==========================================
# ▶️ EXÉCUTION
# ==========================================
if __name__ == '__main__':
    # Plage de test
    b_vals = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    Z_MEAN = 16
    
    # 1. Lancer L=1 (Baseline)
    run_scenario_parallel("S1_L1_Baseline", Z_MEAN, L=1, b_list=b_vals, theta_list=[0.3])
    
    # 2. Lancer L=4 (Ton Modèle)
    run_scenario_parallel("S1_L4_Influence", Z_MEAN, L=4, b_list=b_vals, theta_list=[0.3])
    
    print("\n🏁 TERMINE ! Les fichiers CSV sont dans le dossier 'Resultats_Local'")