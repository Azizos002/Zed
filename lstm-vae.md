# Protocole Industriel : Détection d'Anomalies Réseau Temps Réel (LSTM-VAE & Apache Kafka)

Ce document rassemble l'intégralité du pipeline de déploiement et de validation d'un NIDS (Network Intrusion Detection System) comportemental non-supervisé, calibré pour une infrastructure de type PME / Industrielle.

---

## 🏗️ 1. Architecture Globale du Système

Le système repose sur une architecture découplée de type **Event-Driven Architecture (EDA)** permettant une analyse en mémoire vive (RAM) à haute vélocité sans latence réseau.

[ VM KALI ] -------- Attaques (Light & Heavy) -------> [ VM SURICATA ] (192.168.56.128)
│ (Capture & Edge Extraction)
▼ (Push JSON local)
[ EDGE COLLECTOR ]
│
▼ (Streaming TCP:9092)
[ VM ML ] (192.168.56.130) <--- Apache Kafka <---------------┘
│ (Consumer RAM - Sliding Window 10/10)
▼
[ CONTROLLER PLANE ] ─── Normalisation (Scaler Fixe) ───► [ INFÉRENCE LSTM-VAE (Keras 3) ]
│
▼
[ CRITICAL ALERTS ] ─── (Si MSE > Seuil de Confiance)