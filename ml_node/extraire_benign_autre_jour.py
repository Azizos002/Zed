"""
extraire_benign_autre_jour.py
Extrait uniquement les flux BENIGN d'un autre jour (Tuesday), pour servir
de jeu de test indépendant et vérifier la vraie généralisation du modèle
(pas juste un split train/val venant du même jour).
"""
import pandas as pd
import numpy as np

CHEMIN_ENTREE = "MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv"
CHEMIN_SORTIE = "tuesday_benign_test.csv"

MAPPING_COLONNES = {
    "Total Fwd Packets": "pkts_in",
    "Total Backward Packets": "pkts_out",
    "Total Length of Fwd Packets": "bytes_in",
    "Total Length of Bwd Packets": "bytes_out",
}

df = pd.read_csv(CHEMIN_ENTREE, low_memory=False)
df.columns = df.columns.str.strip()

colonnes = list(MAPPING_COLONNES.keys()) + ["Label"]
df = df[colonnes].rename(columns=MAPPING_COLONNES)

df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df[df["Label"].str.strip() == "BENIGN"]

df.to_csv(CHEMIN_SORTIE, index=False)
print(f"Lignes BENIGN extraites de Tuesday : {len(df)}")
print(f"Sauvegardé dans : {CHEMIN_SORTIE}")
