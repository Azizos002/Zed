"""
enrichir_dataset.py
Fusionne le dataset CIC-IDS-2017 (Monday BENIGN) avec le trafic réel
capturé sur l'infrastructure, pour enrichir l'entraînement du modèle
avec les patterns propres à l'environnement de déploiement (SSH,
Kafka, loopback inter-VM).
"""
import pandas as pd

df_cic = pd.read_csv("monday_features_clean.csv")
df_reel = pd.read_csv("trafic_reel_features.csv")

# On garde uniquement les 4 features communes (le dataset réel n'a pas de Label)
df_cic_features = df_cic[["pkts_in", "pkts_out", "bytes_in", "bytes_out"]]
df_reel_features = df_reel[["pkts_in", "pkts_out", "bytes_in", "bytes_out"]]

df_enrichi = pd.concat([df_cic_features, df_reel_features], ignore_index=True)

print(f"Lignes CIC-IDS-2017 : {len(df_cic_features)}")
print(f"Lignes trafic réel  : {len(df_reel_features)}")
print(f"Total enrichi        : {len(df_enrichi)}")

df_enrichi.to_csv("dataset_enrichi.csv", index=False)
print("Sauvegardé dans dataset_enrichi.csv")
