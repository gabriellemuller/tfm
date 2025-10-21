# -*- coding: utf-8 -*-
"""
Creado el Wed Oct 8 11:37:34 2025

@author: gabri
"""

import random
from collections import deque
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv

# ----------------------- PARÁMETROS -----------------------
N = 200                 # Nodos
network_type = 'ba'     # Tipo de red: 'er' (Erdos-Renyi) o 'ba' (Barabasi-Albert)
z = 6                   # Grado promedio (ER: prob = z/(N-1), BA: m = z//2)
L = 4                   # Número máximo de capas/distancias de vecinos (Largo Alcance)
lam = 0.5               # Lambda (λ) del kernel geométrico (factor de atenuación)
R = 1.0                 # Recompensa R (Coop/Coop)
T_param = 1.6           # Parámetro T (Temptation), modulado por I_i
S = 0.0                 # Pago Sucker (Coop/Defect)
P = 0.0                 # Pago Mutuo Defect (Defect/Defect)
generations = 200       # Número de generaciones/pasos de tiempo
seed = 123              # Semilla para reproducibilidad
theta_global = 0.4      # Umbral θ común para la vigilancia
initial_coop_frac = 0.5 # Fracción inicial de cooperadores
save_plot = True        # Guardar gráfico de series de tiempo
plot_filename = "vigilance_pd_timeseries.png"
save_csv = True         # Guardar datos en CSV
csv_filename = "vigilance_pd_timeseries.csv"
# ----------------------------------------------------------

def build_network(N, network_type, z, seed=None): #Construye la red según el tipo especificado.
    if network_type == 'er':
        p = z / max(1, (N - 1))
        G = nx.erdos_renyi_graph(N, p, seed=seed)
    elif network_type == 'ba':
        m = max(1, z // 2)
        G = nx.barabasi_albert_graph(N, m, seed=seed)
    else:
        raise ValueError("network_type must be 'er' or 'ba'")
        
    # Eliminar nodos aislados (si existen) 
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)
        G = nx.convert_node_labels_to_integers(G)
    return G

def precompute_layers(G, L): #Pre-calcula los conjuntos de vecinos por distancia (capa) para cada nodo.
    layers = {i: [set() for _ in range(L)] for i in G.nodes()}
    for i in G.nodes():
        dist = {i: 0}
        q = deque([i])
        
        while q:
            v = q.popleft()
            d = dist[v]
            if d >= L:
                continue
            
            for nei in G[v]:
                if nei not in dist:
                    dist[nei] = d + 1
                    q.append(nei)
                    
        for node, d in dist.items():
            if 1 <= d <= L:
                layers[i][d-1].add(node)
                
    # Conteo de vecinos por capa (para los denominadores de I_i)
    layer_counts = {i: np.array([len(layers[i][l]) for l in range(L)], dtype=float) for i in G.nodes()}
    return layers, layer_counts

def compute_influence_precomputed(vigilant, layers, layer_counts, alpha): #Calcula el Índice de Influencia Efectiva (I_i) usando pesos pre-calculados (alpha).
    I = {}
    for i, layer_sets in layers.items():
        influence = 0.0
        
        for l_idx, nodeset in enumerate(layer_sets):
            k_l = layer_counts[i][l_idx] # Total de vecinos a distancia l
            
            if k_l > 0:
                m_l = sum(vigilant[n] for n in nodeset) # Vigilantes a distancia l
                f = m_l / k_l # Fracción de vigilantes en la capa l
            else:
                f = 0.0
                
            # Suma ponderada: alpha_norm[l] * f[l]
            influence += alpha[l_idx] * f
            
        I[i] = influence
    return I

def run_simulation(N=N, network_type=network_type, z=z, L=L, lam=lam,
                   R=R, T_param=T_param, S=S, P=P, generations=generations,
                   seed=seed, theta_global=theta_global, initial_coop_frac=initial_coop_frac):
    
    random.seed(seed)
    np.random.seed(seed)

    # 1. Preparación de la Red
    G = build_network(N, network_type, z, seed=seed)
    N_actual = G.number_of_nodes()
    degrees = dict(G.degree())

    # 2. Inicialización de Estados
    strategies = {i: (1 if random.random() < initial_coop_frac else 0) for i in G.nodes()} # 1=C, 0=D
    vigilant = {i: (1 if (strategies[i] == 1 and random.random() < 0.5) else 0) for i in G.nodes()} # 1=Vigilante
    theta = {i: theta_global for i in G.nodes()}

    # 3. Cálculo de Pesos de Influencia (Kernel Geométrico Normalizado)
    alpha = np.array([lam**(l-1) for l in range(1, L+1)], dtype=float)
    
    # Manejar caso de suma cero (para evitar división por cero si L=1 y lam=0, aunque lam=0 y L>1 es el caso de interés)
    alpha_sum = alpha.sum()
    if alpha_sum > 0:
        alpha = alpha / alpha_sum

    # 4. Pre-cálculo de Capas (para optimizar I_i)
    layers, layer_counts = precompute_layers(G, L)

    # Almacenamiento de Series de Tiempo
    history_coop = []
    history_vig = []
    history_avg_T = []

    # Temptation (T_i) inicial antes de que la vigilancia entre en vigor
    T_i = {i: T_param for i in G.nodes()}

    for gen in range(generations):
        # 5. Dinámica de Vigilancia (Actualización I_i y T_i)
        
        # a) Calcular I_i (Influencia de Largo Alcance)
        I = compute_influence_precomputed(vigilant, layers, layer_counts, alpha)

        # b) Actualizar T_i (Temptation modulada)
        for i in G.nodes():
            T_i[i] = R + (T_param - R) * (1 - I[i])

        # c) Actualizar estado de Vigilancia
        for i in G.nodes():
            # Solo los Cooperadores pueden ser Vigilantes, y solo si superan el umbral I_i > theta
            vigilant[i] = 1 if (strategies[i] == 1 and I[i] > theta[i]) else 0

        # 6. Fase de Juego (Dilema del Prisionero)
        
        payoff = {i: 0.0 for i in G.nodes()}
        
        # Jugar PD sobre las aristas de la red
        for i, j in G.edges():
            si, sj = strategies[i], strategies[j]
            
            # Recordar que T_i es individual para los defectores
            if si == 1 and sj == 1:
                payoff[i] += R; payoff[j] += R
            elif si == 1 and sj == 0:
                payoff[i] += S; payoff[j] += T_i[j]  # Defector j usa su T_j
            elif si == 0 and sj == 1:
                payoff[i] += T_i[i]; payoff[j] += S  # Defector i usa su T_i
            else: # si == 0 and sj == 0
                payoff[i] += P; payoff[j] += P

        # 7. Actualización de Estrategias (Imitación Proporcional)
        new_strategies = strategies.copy()
        
        for i in G.nodes():
            neighs = list(G.neighbors(i))
            if not neighs:
                continue
                
            j = random.choice(neighs) # Elegir un vecino para potencialmente imitar
            
            if payoff[j] > payoff[i]:
                # Factor de normalización Phi (del modelo)
                phi = max(degrees[i], degrees[j]) * (max(1.0, T_param) - min(0.0, S)) 
                
                if phi > 0:
                    prob = (payoff[j] - payoff[i]) / phi
                    if random.random() < prob:
                        new_strategies[i] = strategies[j]
        
        strategies = new_strategies

        # Asegurar que los Defectores pierden la vigilancia
        for i in G.nodes():
            if strategies[i] == 0:
                vigilant[i] = 0

        # 8. Recolección de Estadísticas
        frac_coop = sum(strategies.values()) / float(N_actual)
        frac_vig = sum(vigilant.values()) / float(N_actual)
        history_coop.append(frac_coop)
        history_vig.append(frac_vig)
        history_avg_T.append(np.mean(list(T_i.values())))

    return {
        "history_coop": history_coop,
        "history_vig": history_vig,
        "history_avg_T": history_avg_T,
        "final_coop": history_coop[-1],
        "final_vig": history_vig[-1],
        "G": G
    }

def plot_and_save(results, generations, save_plot=True, plot_filename="vigilance_pd_timeseries.png",
                  save_csv=True, csv_filename="vigilance_pd_timeseries.csv", show_plot=True):
    """Genera y guarda el gráfico de series de tiempo y el archivo CSV."""
    
    x = list(range(generations))
    coop = results["history_coop"]
    vig = results["history_vig"]
    avgT = results["history_avg_T"]

    plt.figure(figsize=(10,5))
    plt.plot(x, coop, label="Fracción Cooperadores")
    plt.plot(x, vig, label="Fracción Vigilantes")
    plt.xlabel("Generación")
    plt.ylabel("Fracción")
    plt.ylim(-0.02, 1.02)
    plt.title("Cooperación y Vigilancia en el Tiempo")
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_plot:
        plt.savefig(plot_filename, dpi=150)
        print(f"> Gráfico guardado en {plot_filename}")
    if show_plot:
        plt.show()
    else:
        plt.close()

    if save_csv:
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "frac_coop", "frac_vig", "avg_T"])
            for g, (c, v, t) in enumerate(zip(coop, vig, avgT)):
                writer.writerow([g, c, v, t])
        print(f"> Datos de series de tiempo guardados en {csv_filename}")

if __name__ == "__main__":
    # Ejecutar la simulación con los parámetros globales
    
    print(f"--- SIMULACIÓN DE PRUEBA ---")
    print(f"Parámetros: Red={network_type.upper()}, λ={lam}, T={T_param}, θ={theta_global}")
    
    results = run_simulation(
        N=N, network_type=network_type, z=z, L=L, lam=lam,
        R=R, T_param=T_param, S=S, P=P,
        generations=generations, seed=seed,
        theta_global=theta_global, initial_coop_frac=initial_coop_frac
    )

    print(f"Fracción final de Cooperadores: {results['final_coop']:.3f}")
    print(f"Fracción final de Vigilantes:    {results['final_vig']:.3f}")

    plot_and_save(results, generations, save_plot=save_plot, plot_filename=plot_filename,
                  save_csv=save_csv, csv_filename=csv_filename, show_plot=True)