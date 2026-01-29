# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 21:55:57 2026

@author: gabri
"""

import numpy as np
import networkx as nx
import pandas as pd
import os
import time
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 1. PARAMÈTRES FINAUX (Config "Nuit TFM")
# ==========================================
SEED_GLOBAL = 42
N = 1000
M_BA = 2
THETAS = [0.2, 0.4, 0.6]
L_RANGES = [1, 4]
# Ta sélection stratégique de points b
B_RANGE = [1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
LAMBDA = 0.5
REPETITIONS = 30 
MAX_STEPS = 20000 
CONV_WINDOW = 100

FILE_NAME = "resultats_TFM_FINAL_V1.csv"

def get_neighbors_at_dist(G, node, max_dist):
    path_lengths = nx.single_source_shortest_path_length(G, node, cutoff=max_dist)
    nodes_at_dist = {d: [] for d in range(1, max_dist + 1)}
    for j, d in path_lengths.items():
        if d > 0: nodes_at_dist[d].append(j)
    return nodes_at_dist

def run_single_sim(args):
    corr, theta, L, b, r = args
    np.random.seed(SEED_GLOBAL + r)
    
    G_g = nx.barabasi_albert_graph(N, M_BA, seed=SEED_GLOBAL + r)
    G_i = G_g if corr=="max" else nx.barabasi_albert_graph(N, M_BA, seed=SEED_GLOBAL + r + 1000)
    
    strat = np.random.randint(0, 2, N)
    vig = np.zeros(N)
    for i in range(N):
        if strat[i] == 1 and np.random.rand() < 0.5: vig[i] = 1

    neighborhoods = [get_neighbors_at_dist(G_i, i, L) for i in range(N)]
    alphas = np.array([LAMBDA ** (l - 1) for l in range(1, L + 1)])
    alphas /= alphas.sum()
    
    history_c = []
    for t in range(MAX_STEPS):
        influences = np.zeros(N)
        for i in range(N):
            I_i = 0.0
            for l, neighbors in neighborhoods[i].items():
                if neighbors:
                    f = np.mean([vig[j] for j in neighbors])
                    I_i += alphas[l - 1] * f
            influences[i] = I_i

        new_vig = vig.copy()
        for i in range(N):
            if strat[i] == 0: new_vig[i] = 0
            elif influences[i] >= theta: new_vig[i] = 1
        vig = new_vig

        T_eff = 1.0 + (b - 1.0) * (1.0 - influences)
        payoffs = np.zeros(N)
        for u, v in G_g.edges():
            su, sv = strat[u], strat[v]
            if su == 1 and sv == 1: payoffs[u]+=1; payoffs[v]+=1
            elif su == 1 and sv == 0: payoffs[v]+=T_eff[v]
            elif su == 0 and sv == 1: payoffs[u]+=T_eff[u]

        new_strat = strat.copy()
        for i in range(N):
            neighbors = list(G_g.neighbors(i))
            if neighbors:
                j = np.random.choice(neighbors)
                if payoffs[j] > payoffs[i]:
                    phi = max(G_g.degree(i), G_g.degree(j)) * (max(T_eff[i], T_eff[j]))
                    if phi > 0 and np.random.rand() < (payoffs[j] - payoffs[i]) / phi:
                        new_strat[i] = strat[j]
        strat = new_strat
        
        rho = strat.mean()
        history_c.append(rho)
        if rho == 0: return 0.0, 0.0
        if t > 1000 and t % 100 == 0:
            window = history_c[-CONV_WINDOW:]
            if max(window) - min(window) < 0.001: break
            
    return np.mean(history_c[-CONV_WINDOW:]), np.mean(vig)

if __name__ == "__main__":
    start_total = time.time()
    params = [(c, t, l, b) for c in ["max", "nulle"] for t in THETAS for l in L_RANGES for b in B_RANGE]
    
    for c, t, l, b in params:
        print(f"Lancement : {c} | θ={t} | L={l} | b={b}")
        args_list = [(c, t, l, b, r) for r in range(REPETITIONS)]
        
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(run_single_sim, args_list))
        
        c_vals, v_vals = [r[0] for r in results], [r[1] for r in results]
        res = {"Corr": c, "Theta": t, "L": l, "b": b, "C_mean": np.mean(c_vals), 
               "C_std": np.std(c_vals), "V_mean": np.mean(v_vals), "V_std": np.std(v_vals)}
        
        pd.DataFrame([res]).to_csv(FILE_NAME, mode='a', header=False, index=False)

    print(f"TERMINE en {(time.time()-start_total)/3600:.2f} heures")