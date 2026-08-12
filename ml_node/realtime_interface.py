"""
realtime_interface.py
Sprint 3 - Issue 3.4 : API d'inférence temps réel

Consomme le flux Kafka en continu, maintient un buffer glissant par IP
source, applique le modèle LSTM-VAE, et déclenche une alerte uniquement
après plusieurs dépassements consécutifs du seuil (logique de persistance
pour éviter le bruit de faux positifs isolés).
"""

import os
import json
import time
import logging
from collections import defaultdict, deque

import numpy as np
import joblib
import tensorflow as tf
import keras
from keras import layers, ops
from kafka import KafkaConsumer

from kafka import KafkaProducer

# ============================================================
# CONFIGURATION
# ============================================================
KAFKA_BOOTSTRAP_SERVERS = "192.168.56.130:9092"
KAFKA_TOPIC = "network-features"
KAFKA_TOPIC_ALERTS = "ml-alerts"
KAFKA_GROUP_ID = "realtime-interface"

DOSSIER_MODELE = "modele_fenetre_10_enrichi"
CHEMIN_MODELE = os.path.join(DOSSIER_MODELE, "lstm_candidate.keras")
CHEMIN_SCALER = os.path.join(DOSSIER_MODELE, "global_scaler.pkl")
CHEMIN_METADATA = os.path.join(DOSSIER_MODELE, "metadata.json")

TAILLE_FENETRE = 10
NB_FEATURES = 4
SEUIL_PERSISTANCE = 3  # nombre de dépassements consécutifs avant alerte réelle

FEATURES_ORDRE = ["pkts_in", "pkts_out", "bytes_in", "bytes_out"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("realtime-interface")


# ============================================================
# Couches personnalisées (DOIVENT être identiques à celles utilisées
# lors de l'entraînement dans train_lstm_vae.py, pour permettre à
# Keras de désérialiser correctement le modèle sauvegardé)
# ============================================================
@keras.saving.register_keras_serializable()
class CoucheEchantillonnage(layers.Layer):
    """Reparameterization trick : z = mu + sigma * epsilon"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = keras.random.normal(shape=ops.shape(z_mean))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon


@keras.saving.register_keras_serializable()
class CoucheVAELoss(layers.Layer):
    """Calcule la loss VAE (reconstruction MSE + divergence KL) via add_loss."""
    def call(self, inputs):
        x, x_reconstruit, z_mean, z_log_var = inputs
        loss_reconstruction = ops.mean(ops.square(x - x_reconstruit))
        loss_kl = -0.5 * ops.mean(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
        self.add_loss(loss_reconstruction + loss_kl)
        return x_reconstruit


CUSTOM_OBJECTS = {
    "CoucheEchantillonnage": CoucheEchantillonnage,
    "CoucheVAELoss": CoucheVAELoss,
}


# ============================================================
# Chargement du modèle, du scaler et du seuil (une seule fois)
# ============================================================
def charger_artefacts():
    logger.info("Chargement du modèle, du scaler et du seuil...")

    modele = keras.models.load_model(
        CHEMIN_MODELE,
        custom_objects=CUSTOM_OBJECTS,
        safe_mode=False,
    )
    scaler = joblib.load(CHEMIN_SCALER)

    with open(CHEMIN_METADATA) as f:
        metadata = json.load(f)
    seuil = metadata["seuil_mse"]

    logger.info(f"Modèle chargé. Seuil MSE = {seuil:.6f}")
    return modele, scaler, seuil


@tf.function(reduce_retracing=True)
def inference_compilee(modele, x):
    return modele(x, training=False)


def warm_up(modele):
    """Force la compilation du graphe avant de traiter du vrai trafic."""
    logger.info("Warm-up du modèle...")
    exemple_factice = np.zeros((1, TAILLE_FENETRE, NB_FEATURES), dtype=np.float32)
    _ = inference_compilee(modele, exemple_factice)
    logger.info("Warm-up terminé.")


# ============================================================
# État par IP source (buffer glissant + compteur de persistance)
# ============================================================
buffers_par_ip = defaultdict(lambda: deque(maxlen=TAILLE_FENETRE))
compteurs_depassement = defaultdict(int)


def extraire_features(message: dict):
    """Extrait le vecteur de features dans le bon ordre depuis un message Kafka."""
    try:
        return [float(message[cle]) for cle in FEATURES_ORDRE]
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"Message mal formé, ignoré : {message} ({e})")
        return None


def traiter_message(message: dict, modele, scaler, seuil, producer_alertes):
    src_ip = message.get("src_ip")
    if not src_ip:
        logger.warning(f"Message sans src_ip, ignoré : {message}")
        return

    features = extraire_features(message)
    if features is None:
        return

    buffer = buffers_par_ip[src_ip]
    buffer.append(features)

    # Pas encore assez de flux accumulés pour cette IP
    if len(buffer) < TAILLE_FENETRE:
        return

    # --- Normalisation et inférence ---
    sequence_brute = np.array(buffer, dtype=np.float32)
    sequence_norm = np.clip(scaler.transform(sequence_brute), 0.0, 1.0)
    sequence_norm = sequence_norm.reshape(1, TAILLE_FENETRE, NB_FEATURES).astype(np.float32)

    debut = time.time()
    reconstruction = inference_compilee(modele, sequence_norm)
    latence_ms = (time.time() - debut) * 1000

    mse = float(np.mean(np.square(sequence_norm - np.array(reconstruction))))

    if mse > seuil:
        compteurs_depassement[src_ip] += 1
    else:
        compteurs_depassement[src_ip] = 0

    logger.info(
        f"IP={src_ip} | MSE={mse:.6f} | seuil={seuil:.6f} | "
        f"depassements_consecutifs={compteurs_depassement[src_ip]} | "
        f"latence={latence_ms:.3f}ms"
    )

    # --- Déclenchement de l'alerte (persistance) ---
    if compteurs_depassement[src_ip] >= SEUIL_PERSISTANCE:
        declencher_alerte(src_ip, mse, seuil, latence_ms, producer_alertes)
        compteurs_depassement[src_ip] = 0  # reset après déclenchement


def declencher_alerte(src_ip, mse, seuil, latence_ms, producer_alertes):
    logger.warning(
        f"*** ALERTE CRITIQUE *** IP={src_ip} | MSE={mse:.6f} "
        f"(seuil={seuil:.6f}) | latence_detection={latence_ms:.3f}ms"
    )
    producer_alertes.send(KAFKA_TOPIC_ALERTS, {
        "source": "lstm-vae",
        "src_ip": src_ip,
        "mse": mse,
        "seuil": seuil,
        "timestamp": time.time(),
    })
    producer_alertes.flush()


# ============================================================
# MAIN — boucle de consommation Kafka
# ============================================================
def main():
    modele, scaler, seuil = charger_artefacts()
    warm_up(modele)

    producer_alertes = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    logger.info(f"Connexion au topic Kafka '{KAFKA_TOPIC}'...")
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUP_ID,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
    )
    logger.info("En attente de messages Kafka... (Ctrl+C pour arrêter)")

    try:
        for record in consumer:
            traiter_message(record.value, modele, scaler, seuil, producer_alertes)
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur.")
    finally:
        consumer.close()
        producer_alertes.close()


if __name__ == "__main__":
    main()
