"""
soar.py
Sprint 4 - Orchestrateur SOAR

Consolide les alertes du LSTM-VAE (topic ml-alerts) et de Suricata (topic
suricata-alerts), applique une logique de décision à 3 paliers, vérifie
la liste d'actifs critiques, et déclenche le blocage iptables si besoin.
Les actions de blocage sont idempotentes : une IP déjà bloquée n'est pas
re-bloquée à chaque nouvelle alerte reçue.
"""

import json
import time
import re
import logging
import subprocess

from kafka import KafkaConsumer

KAFKA_BOOTSTRAP_SERVERS = "192.168.56.130:9092"
TOPIC_ML = "ml-alerts"
TOPIC_SURICATA = "suricata-alerts"
FENETRE_CORRELATION_SEC = 30
CHEMIN_ACTIFS_CRITIQUES = "actifs_critiques.json"

IP_VM_EDGE = "192.168.56.128"
CLE_SSH = "/home/aziz/.ssh/soar_key"
UTILISATEUR_EDGE = "aziz"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("soar")

# Historique des dernières alertes par IP, par source (pour la corrélation)
derniere_alerte_ml = {}
derniere_alerte_suricata = {}

# IPs déjà bloquées, pour éviter les actions redondantes (idempotence)
ips_deja_bloquees = set()


def charger_actifs_critiques():
    with open(CHEMIN_ACTIFS_CRITIQUES) as f:
        data = json.load(f)
    return set(data.get("actifs_critiques", []))


def ip_valide(ip: str) -> bool:
    return bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', ip))


def bloquer_ip(ip: str, actifs_critiques: set) -> bool:
    if ip in actifs_critiques:
        logger.warning(f"BLOCAGE REFUSÉ (actif critique) pour IP={ip}")
        return False

    if ip in ips_deja_bloquees:
        logger.info(f"IP={ip} déjà bloquée, action ignorée")
        return False

    if not ip_valide(ip):
        logger.error(f"IP invalide, blocage refusé : {ip}")
        return False

    try:
        subprocess.run(
            [
                "ssh", "-i", CLE_SSH,
                "-o", "StrictHostKeyChecking=no",
                f"{UTILISATEUR_EDGE}@{IP_VM_EDGE}",
                f"sudo iptables -I INPUT -s {ip} -j DROP",
            ],
            check=True, capture_output=True, text=True, timeout=10,
        )
        logger.warning(f"IP BLOQUÉE sur VM-Edge : {ip}")
        ips_deja_bloquees.add(ip)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Échec du blocage distant pour {ip} : {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout lors du blocage distant pour {ip}")
        return False


def evaluer_correlation(src_ip: str, maintenant: float) -> bool:
    """Palier 3 : les deux sources ont alerté sur la même IP dans la fenêtre de corrélation."""
    t_ml = derniere_alerte_ml.get(src_ip)
    t_sur = derniere_alerte_suricata.get(src_ip)
    if t_ml is None or t_sur is None:
        return False
    return abs(t_ml - t_sur) <= FENETRE_CORRELATION_SEC


def traiter_alerte_ml(message: dict, actifs_critiques: set):
    src_ip = message["src_ip"]
    maintenant = time.time()
    derniere_alerte_ml[src_ip] = maintenant

    if evaluer_correlation(src_ip, maintenant):
        logger.warning(f"[PALIER 3] Corrélation ML+Suricata pour IP={src_ip} — priorité HAUTE")
    else:
        logger.info(f"[PALIER 2] Alerte ML seule (persistance confirmée) pour IP={src_ip}")

    bloquer_ip(src_ip, actifs_critiques)


def traiter_alerte_suricata(message: dict, actifs_critiques: set):
    src_ip = message["src_ip"]
    severity = message.get("severity", 3)
    maintenant = time.time()
    derniere_alerte_suricata[src_ip] = maintenant

    if evaluer_correlation(src_ip, maintenant):
        logger.warning(f"[PALIER 3] Corrélation Suricata+ML pour IP={src_ip} — priorité HAUTE")
        bloquer_ip(src_ip, actifs_critiques)
    elif severity <= 2:
        logger.info(f"[PALIER 2] Alerte Suricata sévérité {severity} pour IP={src_ip}")
        bloquer_ip(src_ip, actifs_critiques)
    else:
        logger.info(f"[PALIER 1] Alerte Suricata sévérité {severity} (info) pour IP={src_ip} — log seul")


def main():
    actifs_critiques = charger_actifs_critiques()
    logger.info(f"Actifs critiques chargés : {actifs_critiques}")

    consumer = KafkaConsumer(
        TOPIC_ML, TOPIC_SURICATA,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="soar-orchestrator",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
    )

    logger.info("SOAR en attente d'alertes sur ml-alerts et suricata-alerts...")

    try:
        for record in consumer:
            message = record.value
            if record.topic == TOPIC_ML:
                traiter_alerte_ml(message, actifs_critiques)
            elif record.topic == TOPIC_SURICATA:
                traiter_alerte_suricata(message, actifs_critiques)
    except KeyboardInterrupt:
        logger.info("Arrêt du SOAR.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
