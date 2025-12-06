import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ====================================================================
# --- PARAMÈTRES D'ENTRÉE (À MODIFIER ICI) ----------------------------
# ====================================================================

# CHOISISSEZ VOTRE SCÉNARIO :
NETWORK_TYPE = 'BA'  # 'BA' ou 'ER'
Z_VALUE = 16         # 4 ou 16
L_VALUE = 1         # 1 (Local) ou 4 (Longue Portée)

# ====================================================================
# --- LOGIQUE INTERNE (NE PAS MODIFIER) ------------------------------
# ====================================================================

# Définition des métadonnées de chargement (pour gérer les incohérences CSV)
METADATA = {
    ('BA', 4, 1): {'file': 'rho_vs_b_L1_ba_Z4_mean25.csv', 'sep': ';', 'dec': ',', 'label': 'L=1 (Local)'},
    ('BA', 4, 4): {'file': 'rho_vs_b_L4_ba_Z4_mean25.csv', 'sep': ';', 'dec': ',', 'label': 'L=4 (Longue Portée)'},
    ('ER', 4, 1): {'file': 'rho_vs_b_L1_er_Z4_mean25.csv', 'sep': ';', 'dec': ',', 'label': 'L=1 (Local)'},
    ('ER', 4, 4): {'file': 'rho_vs_b_L4_er_Z4_mean25.csv', 'sep': ';', 'dec': ',', 'label': 'L=4 (Longue Portée)'},
    
    ('BA', 16, 1): {'file': 'rho_vs_b_L1_ba_Z16_mean25.csv', 'sep': ';', 'dec': ',', 'label': 'L=1 (Local)'},
    ('BA', 16, 4): {'file': 'rho_vs_b_L4_ba_Z16_mean25.csv', 'sep': ';', 'dec': ',', 'label': 'L=4 (Longue Portée)'},
    ('ER', 16, 1): {'file': 'rho_vs_b_L1_er_Z16_mean25.csv', 'sep': ';', 'dec': ',', 'label': 'L=1 (Local)'},
    ('ER', 16, 4): {'file': 'rho_vs_b_L4_er_Z16_mean25.csv', 'sep': ';', 'dec': ',', 'label': 'L=4 (Longue Portée)'},
}

def load_data_matrix(network_type, z_value, l_value):
    """Charge les données, nettoie et retourne la matrice RHO (b vs theta) et les étiquettes."""
    
    key = (network_type, z_value, l_value)
    if key not in METADATA:
        print(f"Combinaison de paramètres non trouvée: {key}")
        return None, None, None
    
    meta = METADATA[key]
    
    try:
        # Tente le chargement principal (ajuster ici pour les erreurs persistantes)
        df = pd.read_csv(meta['file'], sep=meta['sep'], decimal=meta['dec'])
    except Exception:
        # Fallback pour le format opposé (au cas où l'encodage est inversé)
        try:
            sep_alt = ',' if meta['sep'] == ';' else ';'
            dec_alt = '.' if meta['dec'] == ',' else ','
            df = pd.read_csv(meta['file'], sep=sep_alt, decimal=dec_alt)
        except Exception:
            print(f"Erreur de chargement irrécupérable pour {meta['file']}")
            return None, None, None
        
    # Nettoyage et normalisation des en-têtes
    df.columns = df.columns.str.strip().str.replace(' ', '')
    
    # Identification des colonnes
    b_col_candidates = [col for col in df.columns if col.startswith('b')]
    if not b_col_candidates:
        print(f"Erreur: Colonne 'b' non trouvée dans {meta['file']}")
        return None, None, None
        
    b_col = b_col_candidates[0]
    theta_cols = [col for col in df.columns if col.startswith('theta_')]
    
    # Conversion des données
    for col in df.columns:
        if col != b_col:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True).str.replace(',', '.'), errors='coerce')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=[b_col]) # Supprimer les lignes vides

    # Préparation de la matrice RHO
    rho_matrix = df[theta_cols].values
    b_labels = df[b_col].values
    theta_labels = [c.replace('theta_', 'θ=') for c in theta_cols]
    
    return rho_matrix, b_labels, theta_labels

# Fonction de tracé
def plot_heatmap(rho_matrix, b_labels, theta_labels, title):
    """Génère une carte de chaleur 2D avec les axes inversés (θ en Y, b en X)."""
    
    # --- TRANSPOSE LA MATRICE ET INVERSE LES LABELS ---
    rho_matrix_T = rho_matrix.T
    
    # Définir la palette de couleurs
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ['#c40000', '#ffff00', '#00c400']) # Rouge -> Jaune -> Vert
    
    # Taille ajustée pour l'inversion
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    # Tracer la matrice transposée (maintenant theta en Y, b en X)
    im = ax.imshow(rho_matrix_T, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    
    # Axes Y (Theta) : Index de la matrice transposée (les colonnes d'origine)
    theta_indices = np.arange(len(theta_labels))
    ax.set_yticks(theta_indices)
    # Utiliser les étiquettes de theta
    ax.set_yticklabels([t.replace('θ=', '') for t in theta_labels])
    ax.set_ylabel("Seuil de Vigilance ($\\theta$)")
    
    # Axes X (b) : Index de la matrice transposée (les lignes d'origine)
    b_indices = np.linspace(0, len(b_labels) - 1, 10, dtype=int)
    ax.set_xticks(b_indices)
    # Utiliser les étiquettes de b
    ax.set_xticklabels([f"{b:.2f}" for b in b_labels[b_indices]], rotation=45, ha="right")
    ax.set_xlabel("Tentation de Défection ($b$)")
    
    # Barre de couleur
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Fraction de Coopérateurs ($\\rho$)', rotation=270, labelpad=15)
    
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    
    filename = f"heatmap_T_vs_B_{NETWORK_TYPE}_Z{Z_VALUE}_L{L_VALUE}.png"
    plt.savefig(filename, dpi=200)
    plt.show()
    print(f"Carte de chaleur sauvegardée sous : {filename}")


if __name__ == '__main__':
    
    # 1. Chargement des données
    rho_matrix, b_labels, theta_labels = load_data_matrix(NETWORK_TYPE, Z_VALUE, L_VALUE)
    
    if rho_matrix is not None and rho_matrix.size > 0:
        # 2. Génération du graphique
        title = f"Diagramme de Phase 2D ({NETWORK_TYPE}, z={Z_VALUE}) | Portée {METADATA[(NETWORK_TYPE, Z_VALUE, L_VALUE)]['label']}"
        plot_heatmap(rho_matrix, b_labels, theta_labels, title)
    else:
        print("\nERREUR: Impossible de générer le graphique. Veuillez vérifier les paramètres ou nettoyer manuellement les fichiers CSV spécifiés.")