# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:34:52 2025

@author: gabri
"""

import pandas as pd
from database_dao import run_query_data

def export_unique_alarms():
    print("⏳ Connexion à la base de données et extraction des alarmes uniques...")
    
    # Cette requête scanne la table pour trouver tous les couples (Code, Message) distincts
    # Elle ignore les dates, on veut juste le catalogue.
    sql_query = r"""
    WITH parsed AS (
        SELECT
            -- Extraction via Regex (Code = groupe 1, Message = groupe 2)
            (regexp_matches(value, '\["([^"]+)","([^"]+)"', 'g'))[1] AS alarm_code,
            (regexp_matches(value, '\["([^"]+)","([^"]+)"', 'g'))[2] AS description
        FROM variable_log_string
        WHERE id_var = 447 -- ID des alarmes
    )
    SELECT DISTINCT
        alarm_code,
        description
    FROM parsed
    ORDER BY alarm_code;
    """
    
    # On lance la requête (sans paramètres de date car on veut tout l'historique)
    df = run_query_data(sql_query, {})
    
    if not df.empty:
        filename = "catalogue_alarmes.csv"
        # Export en CSV compatible Excel (séparateur point-virgule, encodage utf-8-sig pour les accents)
        df.to_csv(filename, index=False, sep=';', encoding='utf-8-sig')
        print(f"✅ Succès ! {len(df)} alarmes uniques trouvées.")
        print(f"📁 Fichier créé : {filename}")
        print("👉 Vous pouvez maintenant ouvrir ce fichier dans Excel pour le classifier.")
    else:
        print("❌ Aucune alarme trouvée ou erreur de connexion.")

if __name__ == "__main__":
    export_unique_alarms()