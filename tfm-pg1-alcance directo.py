import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ----------------------- PARAMETERS -----------------------
N = 300
network_type = 'ba'
z = 6
R = 1.0
S = 0.0
P = 0.0
generations = 200
seed = 123
initial_coop_frac = 0.5
# ----------------------------------------------------------

random.seed(seed)
np.random.seed(seed)

# ----------------------- NETWORK -----------------------
if network_type == 'er':
    p = z / (N - 1)
    G = nx.erdos_renyi_graph(N, p, seed=seed)
elif network_type == 'ba':
    m = max(1, z // 2)
    G = nx.barabasi_albert_graph(N, m, seed=seed)
else:
    raise ValueError("network_type must be 'er' or 'ba'")

# Remove isolated nodes
isolates = list(nx.isolates(G))
if isolates:
    G.remove_nodes_from(isolates)
    G = nx.convert_node_labels_to_integers(G)
    N = G.number_of_nodes()

degrees = dict(G.degree())

# ----------------------- FUNCTION: run_simulation -----------------------
def run_simulation(T, theta):
    # Initialization
    strategies = {i: (1 if random.random() < initial_coop_frac else 0) for i in G.nodes()}
    vigilant = {i: (1 if (strategies[i] == 1 and random.random() < 0.5) else 0) for i in G.nodes()}

    for gen in range(generations):
        # Local vigilance pressure
        I = {}
        for i in G.nodes():
            neighs = list(G[i])
            if not neighs:
                I[i] = 0.0
                continue
            m_i = sum(vigilant[j] for j in neighs)
            I[i] = m_i / len(neighs)

        # Individual temptation
        T_i = {i: R + (T - R) * (1 - I[i]) for i in G.nodes()}

        # Update vigilance
        for i in G.nodes():
            if strategies[i] == 1 and I[i] > theta:
                vigilant[i] = 1
            else:
                vigilant[i] = 0

        # Game phase
        payoff = {i: 0.0 for i in G.nodes()}
        for i, j in G.edges():
            si, sj = strategies[i], strategies[j]
            if si == 1 and sj == 1:
                payoff[i] += R
                payoff[j] += R
            elif si == 1 and sj == 0:
                payoff[i] += S
                payoff[j] += T_i[j]
            elif si == 0 and sj == 1:
                payoff[i] += T_i[i]
                payoff[j] += S
            else:
                payoff[i] += P
                payoff[j] += P

        # Strategy update
        new_strategies = strategies.copy()
        for i in G.nodes():
            neighs = list(G[i])
            if not neighs:
                continue
            j = random.choice(neighs)
            if payoff[j] > payoff[i]:
                phi = max(degrees[i], degrees[j]) * (max(T, 1.0) - min(S, 0.0))
                prob = (payoff[j] - payoff[i]) / phi
                if random.random() < prob:
                    new_strategies[i] = strategies[j]
        strategies = new_strategies

        # Defectors lose vigilance
        for i in G.nodes():
            if strategies[i] == 0:
                vigilant[i] = 0

    # Return final fraction of cooperators
    return sum(strategies.values()) / N


# ----------------------- PARAMETER SWEEP -----------------------
T_values = np.linspace(1.0, 2.0, 21)  # from 1 to 2
theta_values = np.arange(0, 1.01, 0.2)

results = {theta: [] for theta in theta_values}

print("Running simulations...")
for theta in theta_values:
    for T in T_values:
        frac_coop = run_simulation(T, theta)
        results[theta].append(frac_coop)
    print(f"Theta={theta:.1f} done.")

# ----------------------- PLOT -----------------------
plt.figure(figsize=(8,5))
for theta in theta_values:
    plt.plot(T_values, results[theta], marker='o', label=f"θ = {theta:.1f}")

plt.xlabel("Temptation to Defect (b = T)")
plt.ylabel("Final Fraction of Cooperators")
plt.title("Effect of Vigilance Threshold θ on Cooperation Level")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
