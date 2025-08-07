# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:18:06 2025

@author: Gabrielle Muller
"""

import random
import matplotlib.pyplot as plt

#parameters
R = 1     
S = 0     
T = 2 
P = 0    
N = 10    
rounds = 10

#aleatory initiation, c=cooperation, d=defection
agents = {i: random.choice(['C', 'D']) for i in range(N)}
cooperation = []

#DP's simulation
def play_game(s1, s2):
    if s1 == 'C' and s2 == 'C':
        return R, R
    elif s1 == 'C' and s2 == 'D':
        return S, T
    elif s1 == 'D' and s2 == 'C':
        return T, S
    else:
        return P, P

#simulation 
for k in range(rounds):
    #random peers creation
    players = list(agents.keys())
    random.shuffle(players)
    peers = [(players[i], players[i+1]) for i in range(0, N, 2)]

    #payoffs
    payoffs = {i: 0 for i in agents}
    for a, b in peers:
        pa, pb = play_game(agents[a], agents[b])
        payoffs[a] += pa
        payoffs[b] += pb

    #strategy update
    new_agents = agents.copy()
    for i in agents:
        j = random.choice([k for k in agents if k != i])
        if payoffs[j] > payoffs[i]:
            diff = payoffs[j] - payoffs[i]
            phi = (N-1)*(max(1, T) - min(0, S))
            prob = diff / phi
            if random.random() < prob:
                new_agents[i] = agents[j]
    agents = new_agents

    #cooperation rate
    coop_ratio = sum(1 for s in agents.values() if s == 'C') / N
    cooperation.append(coop_ratio)

#resultados
for i, c in enumerate(cooperation, 1):
    print(f"Ronda {i} : Cooperación media = {c:.2f}")
plt.plot(range(1, rounds + 1), cooperation, marker='o', color='green')
plt.xlabel("Ronda")
plt.ylabel("Cooperación media")
plt.title("Evolución de la cooperación entre 10 agentes")
plt.grid(True)
plt.show()
