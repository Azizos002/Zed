"""
extraire_trafic_reel.py
Extrait les features [pkts_in, pkts_out, bytes_in, bytes_out] à partir
d'une capture réelle de trafic normal, pour enrichir/valider le modèle
avec le trafic propre à l'infrastructure (SSH, Kafka, Elasticsearch).
"""
import json
import csv

CHEMIN_ENTREE = "flows_trafic_normal.json"
CHEMIN_SORTIE = "trafic_reel_features.csv"

with open(CHEMIN_ENTREE, "r") as f_in, open(CHEMIN_SORTIE, "w", newline="") as f_out:
    writer = csv.writer(f_out)
    writer.writerow(["pkts_in", "pkts_out", "bytes_in", "bytes_out", "src_ip"])

    nb_lignes = 0
    nb_extraites = 0

    for ligne in f_in:
        nb_lignes += 1
        try:
            evenement = json.loads(ligne)
        except json.JSONDecodeError:
            continue

        flow = evenement.get("flow")
        src_ip = evenement.get("src_ip")
        if not flow or not src_ip:
            continue

        writer.writerow([
            flow.get("pkts_toserver", 0),
            flow.get("pkts_toclient", 0),
            flow.get("bytes_toserver", 0),
            flow.get("bytes_toclient", 0),
            src_ip,
        ])
        nb_extraites += 1

print(f"Lignes lues : {nb_lignes}")
print(f"Flows extraits : {nb_extraites}")
print(f"Fichier de sortie : {CHEMIN_SORTIE}")
