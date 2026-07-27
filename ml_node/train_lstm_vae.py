"""
train_lstm_vae.py
Sprint 3 - Issue 3.2/3.3 : Architecture LSTM-VAE, entraînement, seuil dynamique

Version finale :
- Early Stopping avec restore_best_weights (évite le surapprentissage,
  économise le temps de calcul)
- Mesure de latence correcte via tf.function (chemin d'inférence compilé,
  représentatif de l'usage réel en production)
- Validation de généralisation sur un jour différent (Tuesday BENIGN),
  mesurée via un vrai taux de faux positifs (pas un simple ratio brut)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import keras
from keras import layers, ops, Model
from keras.callbacks import EarlyStopping


# ============================================================
# CONFIGURATION
# ============================================================
TAILLE_FENETRE = 10          # <-- à faire varier pour les tests comparatifs (5, 10, 20)
NB_FEATURES = 4
DIM_LATENTE = 2
UNITES_LSTM = 16
EPOCHS_MAX = 50               # plafond haut, l'Early Stopping coupera avant si besoin
PATIENCE = 3
BATCH_SIZE = 128
RATIO_VALIDATION = 0.2

CHEMIN_DONNEES_TRAIN = "dataset_enrichi.csv"
CHEMIN_DONNEES_TEST_INTER_JOUR = "tuesday_benign_test.csv"
DOSSIER_SORTIE = f"modele_fenetre_{TAILLE_FENETRE}_enrichi"

BORNES_MIN = [0, 0, 0, 0]
BORNES_MAX = [1500, 1500, 1_000_000, 1_000_000]


# ============================================================
# ÉTAPE 1 : Chargement et normalisation
# ============================================================
def charger_et_normaliser(chemin_csv, scaler=None):
    if not os.path.exists(chemin_csv):
        raise FileNotFoundError(f"Fichier introuvable : {chemin_csv}")

    df = pd.read_csv(chemin_csv)
    features = df[["pkts_in", "pkts_out", "bytes_in", "bytes_out"]].values.astype(np.float32)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit([BORNES_MIN, BORNES_MAX])

    features_norm = scaler.transform(features)
    features_norm = np.clip(features_norm, 0.0, 1.0)

    return features_norm, scaler


# ============================================================
# ÉTAPE 2 : Séquences glissantes
# ============================================================
def construire_sequences(data, taille_fenetre):
    nb_sequences = len(data) - taille_fenetre + 1
    if nb_sequences <= 0:
        raise ValueError("Dataset trop petit pour cette taille de fenêtre.")

    sequences = np.zeros((nb_sequences, taille_fenetre, data.shape[1]), dtype=np.float32)
    for i in range(nb_sequences):
        sequences[i] = data[i : i + taille_fenetre]

    return sequences


# ============================================================
# ÉTAPE 3 : Architecture LSTM-VAE
# ============================================================
class CoucheEchantillonnage(layers.Layer):
    """Reparameterization trick : z = mu + sigma * epsilon"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=ops.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


class CoucheVAELoss(layers.Layer):
    """Calcule la loss VAE (reconstruction MSE + divergence KL) via add_loss."""
    def call(self, inputs):
        x, x_reconstruit, z_mean, z_log_var = inputs
        loss_reconstruction = ops.mean(ops.square(x - x_reconstruit))
        loss_kl = -0.5 * ops.mean(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        self.add_loss(loss_reconstruction + loss_kl)
        return x_reconstruit


def construire_modele(taille_fenetre, nb_features, unites_lstm, dim_latente):
    entree = layers.Input(shape=(taille_fenetre, nb_features), name="sequence_entree")

    encode = layers.LSTM(unites_lstm, activation="tanh", name="encodeur_lstm")(entree)
    z_mean = layers.Dense(dim_latente, name="z_mean")(encode)
    z_log_var = layers.Dense(dim_latente, name="z_log_var")(encode)
    z = CoucheEchantillonnage(name="echantillonnage_latent")([z_mean, z_log_var])

    decode = layers.RepeatVector(taille_fenetre, name="repeat_vector")(z)
    decode = layers.LSTM(unites_lstm, activation="tanh", return_sequences=True, name="decodeur_lstm")(decode)
    sortie = layers.TimeDistributed(layers.Dense(nb_features), name="reconstruction")(decode)

    sortie_avec_loss = CoucheVAELoss(name="calcul_loss_vae")([entree, sortie, z_mean, z_log_var])

    modele = Model(entree, sortie_avec_loss, name="lstm_vae")
    modele.compile(optimizer="adam")
    return modele


# ============================================================
# ÉTAPE 4 : Calcul des erreurs MSE (par batch, rapide)
# ============================================================
def calculer_erreurs_mse(modele, sequences, batch_size=512):
    """Calcule l'erreur de reconstruction par séquence, par batch (pas une par une)."""
    reconstructions = modele.predict(sequences, batch_size=batch_size, verbose=0)
    erreurs = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
    return erreurs


def calculer_seuil(erreurs_mse):
    mu = float(np.mean(erreurs_mse))
    sigma = float(np.std(erreurs_mse))
    seuil = mu + 3 * sigma
    return seuil, mu, sigma


def calculer_taux_faux_positifs(erreurs, seuil):
    """Proportion de séquences BENIGN qui dépasseraient le seuil (fausses alertes)."""
    nb_faux_positifs = int(np.sum(erreurs > seuil))
    taux = nb_faux_positifs / len(erreurs) * 100
    return taux, nb_faux_positifs


# ============================================================
# ÉTAPE 5 : Mesure de latence (chemin compilé, représentatif de la prod)
# ============================================================
@tf.function(reduce_retracing=True)
def inference_compilee(modele, x):
    return modele(x, training=False)


def mesurer_latence(modele, exemple_unique, nb_iterations=200):
    # Warm-up : force la compilation tf.function AVANT de mesurer
    _ = inference_compilee(modele, exemple_unique)

    debut = time.time()
    for _ in range(nb_iterations):
        _ = inference_compilee(modele, exemple_unique)
    latence_ms = (time.time() - debut) / nb_iterations * 1000
    return latence_ms


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    print(f"[1/7] Chargement et normalisation (fenêtre={TAILLE_FENETRE})...")
    features_norm, scaler = charger_et_normaliser(CHEMIN_DONNEES_TRAIN)

    print("[2/7] Construction des séquences glissantes...")
    sequences = construire_sequences(features_norm, TAILLE_FENETRE)
    print(f"       -> {sequences.shape[0]} séquences de forme {sequences.shape[1:]}")

    idx_split = int(len(sequences) * (1 - RATIO_VALIDATION))
    sequences_train = sequences[:idx_split]
    sequences_val = sequences[idx_split:]

    print("[3/7] Construction du modèle et entraînement avec Early Stopping...")
    modele = construire_modele(TAILLE_FENETRE, NB_FEATURES, UNITES_LSTM, DIM_LATENTE)

    arbitre = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    debut_entrainement = time.time()
    historique = modele.fit(
        sequences_train, sequences_train,
        validation_data=(sequences_val, sequences_val),
        epochs=EPOCHS_MAX,
        batch_size=BATCH_SIZE,
        callbacks=[arbitre],
        verbose=2,
    )
    duree_entrainement = time.time() - debut_entrainement
    nb_epochs_reelles = len(historique.history["loss"])

    print("[4/7] Calcul du seuil dynamique (mu + 3*sigma) sur validation (même jour)...")
    erreurs_val = calculer_erreurs_mse(modele, sequences_val)
    seuil, mu, sigma = calculer_seuil(erreurs_val)
    print(f"       -> mu={mu:.6f}, sigma={sigma:.6f}, seuil={seuil:.6f}")

    print("[5/7] Vérification de généralisation sur un AUTRE jour (Tuesday BENIGN)...")
    generalisation_ok = os.path.exists(CHEMIN_DONNEES_TEST_INTER_JOUR)
    erreur_moyenne_autre_jour = None
    taux_faux_positifs = None
    if generalisation_ok:
        features_autre_jour, _ = charger_et_normaliser(CHEMIN_DONNEES_TEST_INTER_JOUR, scaler=scaler)
        sequences_autre_jour = construire_sequences(features_autre_jour, TAILLE_FENETRE)
        erreurs_autre_jour = calculer_erreurs_mse(modele, sequences_autre_jour)
        erreur_moyenne_autre_jour = float(np.mean(erreurs_autre_jour))
        taux_faux_positifs, nb_fp = calculer_taux_faux_positifs(erreurs_autre_jour, seuil)
        print(f"       -> erreur moyenne sur Tuesday BENIGN : {erreur_moyenne_autre_jour:.6f}")
        print(f"       -> taux de faux positifs estimé : {taux_faux_positifs:.3f}% ({nb_fp} séquences)")
        if taux_faux_positifs > 5:
            print("       ATTENTION : taux de faux positifs élevé, seuil possiblement trop strict.")
        else:
            print("       OK : taux de faux positifs acceptable.")
    else:
        print(f"       Fichier {CHEMIN_DONNEES_TEST_INTER_JOUR} absent, étape sautée.")

    print("[6/7] Mesure de la latence d'inférence unitaire (chemin compilé)...")
    latence_ms = mesurer_latence(modele, sequences_val[0:1])
    print(f"       -> latence moyenne : {latence_ms:.3f} ms")

    print("[7/7] Sauvegarde des artefacts...")
    modele.save(os.path.join(DOSSIER_SORTIE, "lstm_candidate.keras"))
    joblib.dump(scaler, os.path.join(DOSSIER_SORTIE, "global_scaler.pkl"))

    metadata = {
        "taille_fenetre": TAILLE_FENETRE,
        "unites_lstm": UNITES_LSTM,
        "dim_latente": DIM_LATENTE,
        "epochs_max": EPOCHS_MAX,
        "epochs_reelles_early_stopping": nb_epochs_reelles,
        "patience": PATIENCE,
        "seuil_mse": seuil,
        "mu": mu,
        "sigma": sigma,
        "erreur_moyenne_autre_jour_tuesday": erreur_moyenne_autre_jour,
        "taux_faux_positifs_pourcent": taux_faux_positifs,
        "duree_entrainement_sec": duree_entrainement,
        "latence_inference_ms": latence_ms,
        "loss_finale_train": historique.history["loss"][-1],
        "loss_finale_val": historique.history["val_loss"][-1],
    }
    with open(os.path.join(DOSSIER_SORTIE, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n--- RÉSUMÉ FINAL ---")
    for cle, valeur in metadata.items():
        print(f"{cle}: {valeur}")
    print(f"\nArtefacts sauvegardés dans : {DOSSIER_SORTIE}/")


if __name__ == "__main__":
    main()
