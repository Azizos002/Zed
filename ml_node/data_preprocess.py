"""
data_preprocess.py
Sprint 3 - Issue 3.1 : Intégration et nettoyage du dataset CIC-IDS-2017

Objectif : charger le fichier Monday-WorkingHours (100% trafic BENIGN),
nettoyer les données, extraire notre vecteur de features 4D
[pkts_in, pkts_out, bytes_in, bytes_out], et sauvegarder le résultat
propre pour l'entraînement du modèle LSTM-VAE.
"""

import pandas as pd
import numpy as np
import os

# --- Configuration ---
CHEMIN_CSV = "MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv"
CHEMIN_SORTIE = "monday_features_clean.csv"
TAILLE_CHUNK = 50_000  # lecture par lots pour rester léger en mémoire

# Mapping de nos 4 features vers les colonnes réelles du dataset CIC-IDS-2017
MAPPING_COLONNES = {
    "Total Fwd Packets": "pkts_in",
    "Total Backward Packets": "pkts_out",
    "Total Length of Fwd Packets": "bytes_in",
    "Total Length of Bwd Packets": "bytes_out",
}


def nettoyer_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie un lot (chunk) du dataset :
    - retire les espaces parasites dans les noms de colonnes
    - sélectionne uniquement nos 4 features + le label (pour vérification)
    - remplace les valeurs infinies par NaN, puis supprime les lignes invalides
    """
    # Le dataset CIC-IDS-2017 a des espaces au début des noms de colonnes
    chunk.columns = chunk.columns.str.strip()

    colonnes_requises = list(MAPPING_COLONNES.keys()) + ["Label"]

    # Vérification que toutes les colonnes attendues sont bien présentes
    colonnes_manquantes = [c for c in colonnes_requises if c not in chunk.columns]
    if colonnes_manquantes:
        raise ValueError(f"Colonnes manquantes dans le CSV : {colonnes_manquantes}")

    chunk = chunk[colonnes_requises].copy()

    # Renommage vers nos noms de features standardisés
    chunk = chunk.rename(columns=MAPPING_COLONNES)

    # Le dataset CIC-IDS-2017 contient parfois des valeurs Infinity ou NaN
    # (division par zéro lors du calcul de certains ratios en amont)
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    chunk = chunk.dropna()

    # Sécurité : ne garder que du trafic BENIGN (le fichier Monday est censé
    # l'être à 100%, mais on le vérifie explicitement pour la traçabilité)
    chunk = chunk[chunk["Label"].str.strip() == "BENIGN"]

    return chunk


def main():
    if not os.path.exists(CHEMIN_CSV):
        raise FileNotFoundError(
            f"Le fichier {CHEMIN_CSV} est introuvable. "
            "Vérifie que le dataset a bien été dézippé au bon endroit."
        )

    nb_lignes_totales = 0
    nb_lignes_conservees = 0
    premier_chunk = True

    try:
        # Lecture par chunks pour rester léger en mémoire, même si ce fichier
        # (169 Mo) tiendrait facilement d'un coup sur notre VM (6 Go RAM)
        for chunk in pd.read_csv(CHEMIN_CSV, chunksize=TAILLE_CHUNK, low_memory=False):
            nb_lignes_totales += len(chunk)

            chunk_propre = nettoyer_chunk(chunk)
            nb_lignes_conservees += len(chunk_propre)

            # Écriture incrémentale dans le fichier de sortie
            chunk_propre.to_csv(
                CHEMIN_SORTIE,
                mode="w" if premier_chunk else "a",
                header=premier_chunk,
                index=False,
            )
            premier_chunk = False

    except pd.errors.ParserError as e:
        print(f"Erreur de parsing du CSV : {e}")
        raise
    except Exception as e:
        print(f"Erreur inattendue pendant le traitement : {e}")
        raise

    print("--- Résumé du preprocessing ---")
    print(f"Lignes lues au total       : {nb_lignes_totales}")
    print(f"Lignes conservées (propres): {nb_lignes_conservees}")
    print(f"Lignes rejetées (NaN/Inf)  : {nb_lignes_totales - nb_lignes_conservees}")
    print(f"Fichier de sortie          : {CHEMIN_SORTIE}")


if __name__ == "__main__":
    main()
