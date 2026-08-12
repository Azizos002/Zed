"""
train_lstm_vae.py
Sprint 3 - Issue 3.2/3.3 : Architecture LSTM-VAE, entraînement, seuil dynamique

Charge le dataset nettoyé (monday_features_clean.csv), construit les séquences
temporelles glissantes, normalise, entraîne un LSTM-VAE non-supervisé, calcule
le seuil de détection dynamique (mu + 3*sigma), et sauvegarde tous les artefacts.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

import keras
from keras import layers, ops, Model


# ============================================================
# CONFIGURATION (paramètres à faire varier pour les tests comparatifs)
# ============================================================
TAILLE_FENETRE = 10          # <-- à changer pour tester 5, 10, 20...
NB_FEATURES = 4               # pkts_in, pkts_out, bytes_in, bytes_out
DIM_LATENTE = 2
UNITES_LSTM = 16
EPOCHS = 20
BATCH_SIZE = 128
RATIO_VALIDATION = 0.2

CHEMIN_DONNEES = "monday_features_clean.csv"
DOSSIER_SORTIE = f"modele_fenetre_{TAILLE_FENETRE}"

# Bornes fixes du MinMaxScaler (rappel cahier des charges : saturation
# volontaire pour les attaques volumétriques, pas de scaler dynamique)
BORNES_MIN = [0, 0, 0, 0]
BORNES_MAX = [1500, 1500, 1_000_000, 1_000_000]


# ============================================================
# ÉTAPE 1 : Chargement et normalisation
# ============================================================
def charger_et_normaliser(chemin_csv):
    if not os.path.exists(chemin_csv):
        raise FileNotFoundError(f"Fichier introuvable : {chemin_csv}")

    df = pd.read_csv(chemin_csv)
    features = df[["pkts_in", "pkts_out", "bytes_in", "bytes_out"]].values.astype(np.float32)

    scaler = MinMaxScaler()
    scaler.fit([BORNES_MIN, BORNES_MAX])  # bornes fixes, pas fit sur les données réelles
    features_norm = scaler.transform(features)
    features_norm = np.clip(features_norm, 0.0, 1.0)  # sécurité anti-dépassement

    return features_norm, scaler


# ============================================================
# ÉTAPE 2 : Construction des séquences (fenêtre glissante)
# ============================================================
def construire_sequences(data, taille_fenetre):
    """
    Transforme un tableau (N, F) en séquences (N - taille_fenetre + 1, taille_fenetre, F)
    via une fenêtre glissante, pour donner une mémoire temporelle au modèle.
    """
    nb_sequences = len(data) - taille_fenetre + 1
    if nb_sequences <= 0:
        raise ValueError("Le dataset est trop petit pour la taille de fenêtre demandée.")

    sequences = np.zeros((nb_sequences, taille_fenetre, data.shape[1]), dtype=np.float32)
    for i in range(nb_sequences):
        sequences[i] = data[i : i + taille_fenetre]

    return sequences


# ============================================================
# ÉTAPE 3 : Architecture LSTM-VAE (Keras 3)
# ============================================================
class CoucheEchantillonnage(layers.Layer):
    """Reparameterization trick : z = mu + sigma * epsilon"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=ops.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class CoucheVAELoss(layers.Layer):
    """
    Calcule la loss VAE (reconstruction MSE + divergence KL) et l'ajoute
    au graphe via add_loss, car Keras 3 n'autorise plus .add_loss() direct
    sur un modèle fonctionnel de la même façon qu'avant.
    """
    def call(self, inputs):
        x, x_reconstruit, z_mean, z_log_var = inputs

        loss_reconstruction = ops.mean(ops.square(x - x_reconstruit))
        loss_kl = -0.5 * ops.mean(
            1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var)
        )
        self.add_loss(loss_reconstruction + loss_kl)
        return x_reconstruit


def construire_modele(taille_fenetre, nb_features, unites_lstm, dim_latente):
    entree = layers.Input(shape=(taille_fenetre, nb_features), name="sequence_entree")

    # --- Encodeur ---
    encode = layers.LSTM(unites_lstm, activation="tanh", name="encodeur_lstm")(entree)
    z_mean = layers.Dense(dim_latente, name="z_mean")(encode)
    z_log_var = layers.Dense(dim_latente, name="z_log_var")(encode)
    z = CoucheEchantillonnage(name="echantillonnage_latent")([z_mean, z_log_var])

    # --- Décodeur ---
    decode = layers.RepeatVector(taille_fenetre, name="repeat_vector")(z)
    decode = layers.LSTM(unites_lstm, activation="tanh", return_sequences=True, name="decodeur_lstm")(decode)
    sortie = layers.TimeDistributed(layers.Dense(nb_features), name="reconstruction")(decode)

    sortie_avec_loss = CoucheVAELoss(name="calcul_loss_vae")([entree, sortie, z_mean, z_log_var])

    modele = Model(entree, sortie_avec_loss, name="lstm_vae")
    modele.compile(optimizer="adam")
    return modele


# ============================================================
# ÉTAPE 4 : Calcul du seuil dynamique (mu + 3*sigma)
# ============================================================
def calculer_seuil(modele, sequences_validation):
    reconstructions = modele.predict(sequences_validation, verbose=0)
    erreurs_mse = np.mean(np.square(sequences_validation - reconstructions), axis=(1, 2))

    mu = float(np.mean(erreurs_mse))
    sigma = float(np.std(erreurs_mse))
    seuil = mu + 3 * sigma

    return seuil, mu, sigma, erreurs_mse


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    print(f"[1/5] Chargement et normalisation (fenêtre={TAILLE_FENETRE})...")
    features_norm, scaler = charger_et_normaliser(CHEMIN_DONNEES)

    print("[2/5] Construction des séquences glissantes...")
    sequences = construire_sequences(features_norm, TAILLE_FENETRE)
    print(f"       -> {sequences.shape[0]} séquences de forme {sequences.shape[1:]}")

    # Split train/validation (données déjà en ordre temporel, split simple)
    idx_split = int(len(sequences) * (1 - RATIO_VALIDATION))
    sequences_train = sequences[:idx_split]
    sequences_val = sequences[idx_split:]

    print("[3/5] Construction et entraînement du modèle LSTM-VAE...")
    modele = construire_modele(TAILLE_FENETRE, NB_FEATURES, UNITES_LSTM, DIM_LATENTE)

    debut_entrainement = time.time()
    historique = modele.fit(
        sequences_train, sequences_train,
        validation_data=(sequences_val, sequences_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )
    duree_entrainement = time.time() - debut_entrainement

    print("[4/5] Calcul du seuil dynamique (mu + 3*sigma) sur le jeu de validation...")
    seuil, mu, sigma, erreurs_mse = calculer_seuil(modele, sequences_val)
    print(f"       -> mu={mu:.6f}, sigma={sigma:.6f}, seuil={seuil:.6f}")

    print("[5/5] Mesure de la latence d'inférence unitaire...")
    exemple_unique = sequences_val[0:1]
    modele.predict(exemple_unique, verbose=0)  # warm-up
    debut_inference = time.time()
    for _ in range(100):
        modele.predict(exemple_unique, verbose=0)
    latence_moyenne_ms = (time.time() - debut_inference) / 100 * 1000

    # --- Sauvegarde des artefacts ---
    modele.save(os.path.join(DOSSIER_SORTIE, "lstm_candidate.keras"))
    joblib.dump(scaler, os.path.join(DOSSIER_SORTIE, "global_scaler.pkl"))

    metadata = {
        "taille_fenetre": TAILLE_FENETRE,
        "unites_lstm": UNITES_LSTM,
        "dim_latente": DIM_LATENTE,
        "epochs": EPOCHS,
        "seuil_mse": seuil,
        "mu": mu,
        "sigma": sigma,
        "duree_entrainement_sec": duree_entrainement,
        "latence_inference_ms": latence_moyenne_ms,
        "loss_finale_train": historique.history["loss"][-1],
        "loss_finale_val": historique.history["val_loss"][-1],
    }
    with open(os.path.join(DOSSIER_SORTIE, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n--- RÉSUMÉ ---")
    for cle, valeur in metadata.items():
        print(f"{cle}: {valeur}")
    print(f"\nArtefacts sauvegardés dans : {DOSSIER_SORTIE}/")


if __name__ == "__main__":
    main()