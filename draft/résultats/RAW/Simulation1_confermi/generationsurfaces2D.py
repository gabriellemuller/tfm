import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ====================================================================
# --- 1. METADATA CORRIGÉE (Noms de fichiers sans '_') ---------------
# ====================================================================
METADATA = {
    # Scénarios Z=4
    'BA_Z4_L1': {'file': 'rho_vs_b_L1ba_Z4_mean25.csv', 'sep': ',', 'dec': '.'},
    'BA_Z4_L4': {'file': 'rho_vs_b_L4ba_Z4_mean25.csv', 'sep': ',', 'dec': '.'},
    'ER_Z4_L1': {'file': 'rho_vs_b_L1er_Z4_mean25.csv', 'sep': ',', 'dec': '.'},
    'ER_Z4_L4': {'file': 'rho_vs_b_L4er_Z4_mean25.csv', 'sep': ',', 'dec': '.'},
    
    # Scénarios Z=16
    'BA_Z16_L1': {'file': 'rho_vs_b_L1ba_Z16_mean25.csv', 'sep': ',', 'dec': '.'},
    'BA_Z16_L4': {'file': 'rho_vs_b_L4ba_Z16_mean25.csv', 'sep': ',', 'dec': '.'},
    'ER_Z16_L1': {'file': 'rho_vs_b_L1er_Z16_mean25.csv', 'sep': ',', 'dec': '.'},
    'ER_Z16_L4': {'file': 'rho_vs_b_L4er_Z16_mean25.csv', 'sep': ',', 'dec': '.'},
}

def load_matrix(key):
    """Charge et nettoie une matrice rho avec gestion robuste des séparateurs."""
    
    if key not in METADATA:
        print(f"ERREUR : Clé '{key}' inconnue.")
        return None, None, None

    meta = METADATA[key]
    filename = meta['file']
    
    # Tentative 1 : Paramètres par défaut (souvent point-virgule/virgule pour vos fichiers)
    try:
        # On essaie d'abord le format européen souvent vu dans vos fichiers
        df = pd.read_csv(filename, sep=';', decimal=',')
        if len(df.columns) < 2: # Si le séparateur n'a pas marché
             raise ValueError("Mauvais séparateur")
    except:
        # Tentative 2 : Format Anglo-saxon
        try:
            df = pd.read_csv(filename, sep=',', decimal='.')
        except Exception as e:
            print(f"Erreur fatale de chargement pour {filename}: {e}")
            return None, None, None
        
    # Nettoyage des colonnes
    df.columns = df.columns.str.strip().str.replace(' ', '')
    
    # Identification des colonnes
    b_col_list = [c for c in df.columns if c.startswith('b')]
    theta_col_list = [c for c in df.columns if c.startswith('theta')]
    
    if not b_col_list or not theta_col_list:
        print(f"Erreur de structure dans {filename} (Colonnes manquantes)")
        return None, None, None
        
    b_col = b_col_list[0]
    
    # Conversion Numérique (Sécurité supplémentaire)
    for col in df.columns:
        if df[col].dtype == object:
             # On remplace les virgules par des points pour la conversion float
             df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        else:
             df[col] = pd.to_numeric(df[col], errors='coerce')
             
    df = df.dropna(subset=[b_col])
    
    return df[theta_col_list].values, df[b_col].values, theta_col_list

def plot_diff_map(ax, diff_matrix, b_labels, theta_labels, title):
    """Trace la différence avec Axes Inversés (Theta en Y)."""
    
    if diff_matrix is None:
        ax.text(0.5, 0.5, "Données Manquantes", ha='center', va='center')
        return

    # Transposition pour avoir Theta en Y
    diff_matrix_T = diff_matrix.T
    
    # Échelle de couleur symétrique (pour bien voir le gain vs perte)
    limit = np.nanmax(np.abs(diff_matrix))
    limit = max(0.1, limit) if not np.isnan(limit) else 0.1
    
    im = ax.imshow(diff_matrix_T, cmap='seismic_r', aspect='auto', interpolation='nearest', 
                   vmin=-limit, vmax=limit)
    
    # Axes Y (Theta)
    ax.set_yticks(np.arange(len(theta_labels)))
    ax.set_yticklabels([t.replace('theta_', '') for t in theta_labels], fontsize=8)
    ax.set_ylabel("Seuil ($\\theta$)")
    
    # Axes X (b)
    b_idx = np.linspace(0, len(b_labels)-1, 6, dtype=int)
    ax.set_xticks(b_idx)
    ax.set_xticklabels([f"{b:.1f}" for b in b_labels[b_idx]], fontsize=8)
    ax.set_xlabel("Tentation ($b$)")
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    return im

# --- EXÉCUTION ---
if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    scenarios = [('BA', 'Z4'), ('ER', 'Z4'), ('BA', 'Z16'), ('ER', 'Z16')]
    
    # Variable pour la barre de couleur
    im_ref = None 

    for i, (net, z) in enumerate(scenarios):
        # Construction des clés correspondant aux clés du dictionnaire METADATA
        key1 = f"{net}_{z}_L1"
        key4 = f"{net}_{z}_L4"
        
        print(f"Traitement {key1} vs {key4}...")
        
        # Chargement
        mat1, b1, t1 = load_matrix(key1)
        mat4, b4, t4 = load_matrix(key4)
        
        if mat1 is not None and mat4 is not None:
            # S'assurer que les dimensions correspondent (parfois les simulations s'arrêtent avant)
            r = min(mat1.shape[0], mat4.shape[0])
            c = min(mat1.shape[1], mat4.shape[1])
            
            # Calcul de la différence : L4 (Nouveau) - L1 (Ancien)
            diff = mat4[:r, :c] - mat1[:r, :c]
            
            # Utiliser les labels du premier fichier valide
            im_ref = plot_diff_map(axes[i], diff, b1[:r], t1[:c], f"Gain {net} {z} (L4 - L1)")
        else:
            axes[i].text(0.5, 0.5, "Erreur Chargement", ha='center')
            axes[i].set_title(f"Gain {net} {z}")

    # Barre de couleur commune
    if im_ref:
        cbar = fig.colorbar(im_ref, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
        cbar.set_label('Différence de Coopération ($\\Delta \\rho$)\nBleu = Gain | Rouge = Perte', rotation=270, labelpad=20)

    plt.suptitle("Cartes Différentielles (L4 - L1) : Impact de l'Influence à Longue Portée", fontsize=16)
    plt.savefig("diff_maps_L4_L1_final.png", dpi=150)
    plt.show()
    print("Graphique généré : diff_maps_L4_L1_final.png")