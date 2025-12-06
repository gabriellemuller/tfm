# -*- coding: utf-8 -*-
"""
Modèle de Coopération sous Vigilance Étendue (Longue Portée)
VERSION AVEC COÛT (COST_PARAM)

Ce module est une extension de tfm_pg1_confermi.py.
Il introduit un coût pour les agents vigilants.
"""

import random
from collections import deque
import numpy as np
import networkx as nx

# =============================================================================
# FONCTIONS UTILITAIRES (Identiques à la version de base)
# =============================================================================

def build_network(N, network_type, z, seed=None):
    if network_type == 'er':
        p = z / max(1, (N - 1))
        G = nx.erdos_renyi_graph(N, p, seed=seed)
    elif network_type == 'ba':
        m = max(1, z // 2)
        G = nx.barabasi_albert_graph(N, m, seed=seed)
    else:
        raise ValueError("Type de réseau inconnu. Utilisez 'er' ou 'ba'.")
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)
        G = nx.convert_node_labels_to_integers(G)
    return G

def precompute_layers(G, L):
    layers = {i: [set() for _ in range(L)] for i in G.nodes()}
    for i in G.nodes():
        dist = {i: 0}
        q = deque([i])
        while q:
            v = q.popleft()
            d = dist[v]
            if d >= L: continue
            for nei in G[v]:
                if nei not in dist:
                    dist[nei] = d + 1
                    q.append(nei)
        for node, d in dist.items():
            if 1 <= d <= L:
                layers[i][d-1].add(node)
    layer_counts = {i: np.array([len(layers[i][l]) for l in range(L)], dtype=float) for i in G.nodes()}
    return layers, layer_counts

def compute_influence_precomputed(vigilant, layers, layer_counts, alpha):
    I = {}
    for i, layer_sets in layers.items():
        influence = 0.0
        for l_idx, nodeset in enumerate(layer_sets):
            k_l = layer_counts[i][l_idx]
            if k_l > 0:
                m_l = sum(vigilant[n] for n in nodeset)
                fraction = m_l / k_l
            else:
                fraction = 0.0
            influence += alpha[l_idx] * fraction
        I[i] = influence
    return I

# =============================================================================
# MOTEUR DE SIMULATION (MODIFIÉ POUR LE COÛT)
# =============================================================================

def run_simulation(N=200, network_type='ba', z=4, L=4, lam=0.5,
                   R=1.0, T_param=1.5, S=0.0, P=0.0, 
                   generations=200, seed=None, 
                   theta_global=0.4, initial_coop_frac=0.5, 
                   K_FERMI=0.1, 
                   COST_PARAM=0.0): # <--- Paramètre ajouté ici
    
    random.seed(seed)
    np.random.seed(seed)

    # 1. Construction
    G = build_network(N, network_type, z, seed=seed)
    N_actual = G.number_of_nodes()
    
    # 2. Pré-calculs
    alpha = np.array([lam**(l-1) for l in range(1, L+1)], dtype=float)
    if alpha.sum() > 0: alpha /= alpha.sum()
    layers, layer_counts = precompute_layers(G, L)

    # 3. Init
    strategies = {i: (1 if random.random() < initial_coop_frac else 0) for i in G.nodes()}
    vigilant = {i: (1 if (strategies[i] == 1 and random.random() < 0.5) else 0) for i in G.nodes()}
    theta = {i: theta_global for i in G.nodes()}
    T_i = {i: T_param for i in G.nodes()}

    history_coop = []
    history_vig = []

    # Calcul du coût absolu (c = delta * R)
    cost_value = COST_PARAM * R

    # --- BOUCLE ---
    for gen in range(generations):
        
        # A. Vigilance
        I = compute_influence_precomputed(vigilant, layers, layer_counts, alpha)

        for i in G.nodes():
            T_i[i] = R + (T_param - R) * (1 - I[i]) 

        for i in G.nodes():
            if strategies[i] == 1 and I[i] > theta[i]:
                vigilant[i] = 1
            else:
                vigilant[i] = 0

        # B. Payoffs Bruts
        payoff = {i: 0.0 for i in G.nodes()}
        for i, j in G.edges():
            si, sj = strategies[i], strategies[j]
            if si == 1 and sj == 1: payoff[i] += R; payoff[j] += R
            elif si == 1 and sj == 0: payoff[i] += S; payoff[j] += T_i[j]
            elif si == 0 and sj == 1: payoff[i] += T_i[i]; payoff[j] += S
            else: payoff[i] += P; payoff[j] += P

        # C. APPLICATION DU COÛT (La partie spécifique à ce fichier)
        if cost_value > 0:
            for i in G.nodes():
                if vigilant[i] == 1:
                    payoff[i] -= cost_value # On réduit le gain net du vigilant

        # D. Stratégies (Fermi)
        new_strategies = strategies.copy()
        nodes_list = list(G.nodes())
        
        for i in nodes_list:
            neighs = list(G.neighbors(i))
            if not neighs: continue
            j = random.choice(neighs)
            
            delta_payoff = payoff[j] - payoff[i]
            
            try:
                prob_fermi = 1 / (1 + np.exp(-delta_payoff / K_FERMI))
            except OverflowError:
                prob_fermi = 0.0 if -delta_payoff / K_FERMI > 0 else 1.0
            
            if random.random() < prob_fermi:
                new_strategies[i] = strategies[j]
        
        strategies = new_strategies

        # Un défecteur ne peut pas être vigilant
        for i in G.nodes():
            if strategies[i] == 0: vigilant[i] = 0

        # E. Stats
        frac_coop = sum(strategies.values()) / float(N_actual)
        frac_vig = sum(vigilant.values()) / float(N_actual)
        history_coop.append(frac_coop)
        history_vig.append(frac_vig)

    return {
        "history_coop": history_coop,
        "history_vig": history_vig,
        "final_coop": history_coop[-1] if history_coop else 0.0,
        "final_vig": history_vig[-1] if history_vig else 0.0
    }