"""
valider_trafic_reel.py
Mesure le taux de faux positifs du modèle LSTM-VAE actuel sur un
échantillon de trafic réel de l'infrastructure (SSH, Kafka, loopback).
"""
import json
import numpy as np
import pandas as pd
import joblib
import keras
from keras import layers, ops

DOSSIER_MODELE = "modele_fenetre_10_enrichi"
TAILLE_FENETRE = 10

@keras.saving.register_keras_serializable()
class CoucheEchantillonnage(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=ops.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

@keras.saving.register_keras_serializable()
class CoucheVAELoss(layers.Layer):
    def call(self, inputs):
        x, x_reconstruit, z_mean, z_log_var = inputs
        loss_reconstruction = ops.mean(ops.square(x - x_reconstruit))
        loss_kl = -0.5 * ops.mean(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        self.add_loss(loss_reconstruction + loss_kl)
        return x_reconstruit

modele = keras.models.load_model(
    f"{DOSSIER_MODELE}/lstm_candidate.keras",
    custom_objects={"CoucheEchantillonnage": CoucheEchantillonnage, "CoucheVAELoss": CoucheVAELoss},
    safe_mode=False,
)
scaler = joblib.load(f"{DOSSIER_MODELE}/global_scaler.pkl")
with open(f"{DOSSIER_MODELE}/metadata.json") as f:
    seuil = json.load(f)["seuil_mse"]

df = pd.read_csv("trafic_reel_features.csv")
print(f"Total de flows réels chargés : {len(df)}")
print(df["src_ip"].value_counts())

resultats_par_ip = []

for ip, groupe in df.groupby("src_ip"):
    features = groupe[["pkts_in", "pkts_out", "bytes_in", "bytes_out"]].values.astype(np.float32)
    if len(features) < TAILLE_FENETRE:
        print(f"IP={ip} : seulement {len(features)} flows, ignorée (minimum {TAILLE_FENETRE})")
        continue

    features_norm = np.clip(scaler.transform(features), 0.0, 1.0)

    nb_sequences = len(features_norm) - TAILLE_FENETRE + 1
    sequences = np.zeros((nb_sequences, TAILLE_FENETRE, 4), dtype=np.float32)
    for i in range(nb_sequences):
        sequences[i] = features_norm[i:i + TAILLE_FENETRE]

    reconstructions = modele.predict(sequences, verbose=0)
    erreurs = np.mean(np.square(sequences - reconstructions), axis=(1, 2))

    nb_faux_positifs = int(np.sum(erreurs > seuil))
    taux = nb_faux_positifs / len(erreurs) * 100

    resultats_par_ip.append((ip, len(erreurs), nb_faux_positifs, taux))
    print(f"IP={ip} | séquences={len(erreurs)} | faux positifs={nb_faux_positifs} ({taux:.2f}%) | erreur moyenne={np.mean(erreurs):.6f}")

print("\n--- RÉSUMÉ GLOBAL ---")
total_sequences = sum(r[1] for r in resultats_par_ip)
total_fp = sum(r[2] for r in resultats_par_ip)
if total_sequences > 0:
    print(f"Taux de faux positifs global : {total_fp / total_sequences * 100:.2f}% ({total_fp}/{total_sequences})")
else:
    print("Pas assez de données pour calculer un taux global.")
