"""
edge_collector.py
Producer Kafka temps réel : lit eve.json en continu (comme un `tail -f`),
extrait les 4 features volumétriques par flux, et les envoie immédiatement
vers Kafka - un message par flux, sans attendre d'en accumuler plusieurs
(c'est realtime_interface.py, côté VM-ML, qui gère l'accumulation).
"""

import json
import time
import logging

from kafka import KafkaProducer


# ============================================================
# CONFIGURATION
# ============================================================
CHEMIN_EVE_JSON = "/var/log/suricata/eve.json"
KAFKA_BOOTSTRAP_SERVERS = "192.168.56.130:9092"
KAFKA_TOPIC = "network-features"
KAFKA_TOPIC_ALERTS = "suricata-alerts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("edge-collector")


def extraire_features(evenement: dict):
    """
    Extrait les 4 features volumétriques d'un événement EVE JSON de type
    'flow'. Retourne None si l'événement n'est pas exploitable (pas un
    flow, ou champs manquants).
    """
    if evenement.get("event_type") != "flow":
        return None

    flow = evenement.get("flow")
    if not flow:
        return None

    src_ip = evenement.get("src_ip")
    if not src_ip:
        return None

    try:
        return {
            "src_ip": src_ip,
            "pkts_in": flow.get("pkts_toserver", 0),
            "pkts_out": flow.get("pkts_toclient", 0),
            "bytes_in": flow.get("bytes_toserver", 0),
            "bytes_out": flow.get("bytes_toclient", 0),
            "timestamp": time.time(),
        }
    except (TypeError, ValueError) as e:
        logger.warning(f"Événement mal formé, ignoré : {e}")
        return None

def extraire_alerte(evenement: dict):
    """Extrait les infos utiles d'un événement EVE JSON de type 'alert'."""
    if evenement.get("event_type") != "alert":
        return None

    alert = evenement.get("alert")
    src_ip = evenement.get("src_ip")
    if not alert or not src_ip:
        return None

    return {
        "source": "suricata",
        "src_ip": src_ip,
        "signature": alert.get("signature", ""),
        "severity": alert.get("severity", 3),
        "category": alert.get("category", ""),
        "timestamp": time.time(),
    }

def suivre_fichier(chemin_fichier):
    """
    Génère les nouvelles lignes d'un fichier en continu, comme `tail -f`.
    Utilise f.seek(0, 2) pour se placer à la fin du fichier au démarrage,
    puis lit uniquement les nouvelles lignes ajoutées.
    """
    with open(chemin_fichier, "r") as f:
        f.seek(0, 2)  # se placer à la fin du fichier
        while True:
            ligne = f.readline()
            if not ligne:
                time.sleep(0.1)
                continue
            yield ligne


def main():
    logger.info(f"Connexion au broker Kafka {KAFKA_BOOTSTRAP_SERVERS}...")
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=0,  # envoi immédiat, pas de mise en batch artificielle (latence)
    )
    logger.info(f"Suivi du fichier {CHEMIN_EVE_JSON} en continu...")

    nb_envoyes = 0
    nb_ignores = 0

    try:
        for ligne in suivre_fichier(CHEMIN_EVE_JSON):
            try:
                evenement = json.loads(ligne)
            except json.JSONDecodeError:
                nb_ignores += 1
                continue

            features = extraire_features(evenement)
            if features is not None:
                producer.send(KAFKA_TOPIC, features)
                nb_envoyes += 1

            alerte = extraire_alerte(evenement)
            if alerte is not None:
                producer.send(KAFKA_TOPIC_ALERTS, alerte)
                logger.info(f"Alerte Suricata transmise : {alerte['signature']} (IP={alerte['src_ip']})")

            if features is None and alerte is None:
                nb_ignores += 1

            if nb_envoyes % 100 == 0 and nb_envoyes > 0:
                logger.info(f"Envoyés: {nb_envoyes} | Ignorés: {nb_ignores}")

    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur.")
    finally:
        producer.flush()
        producer.close()
        logger.info(f"Total final -> Envoyés: {nb_envoyes} | Ignorés: {nb_ignores}")


if __name__ == "__main__":
    main()
