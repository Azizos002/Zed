# Issues GitHub Projects — Prêtes à copier-coller

Pour chaque issue : Titre → copier dans le champ titre. Corps → copier dans la description.
Label epic + champ Sprint à assigner une fois l'issue créée et ajoutée au Project.

---

## SPRINT 1 — Déploiement Suricata (label: epic-detection)

### Issue 1.1
**Titre:** [Sprint 1] 1.1 - Capture réseau en mode promiscuous
**Corps:**
En tant que Système, je dois capturer le trafic réseau via l'interface en mode promiscuous sans altérer la latence de production.

**Livrable attendu:** Interface réseau en mode promiscuous fonctionnelle, capture de paquets validée.

---

### Issue 1.2
**Titre:** [Sprint 1] 1.2 - Configuration AF_PACKET (multithreading)
**Corps:**
En tant qu'Architecte, je veux configurer Suricata pour activer le multithreading (AF_PACKET) et éviter le packet drop.

**Livrable attendu:** suricata.yaml configuré en AF_PACKET, cluster_flow activé, zéro packet drop observé.

---

### Issue 1.3
**Titre:** [Sprint 1] 1.3 - Sérialisation EVE JSON
**Corps:**
En tant que Système, je dois sérialiser les flux décodés au format unifié EVE JSON pour faciliter l'ingestion Big Data.

**Livrable attendu:** Fichier eve.json généré en continu, structure validée (capture d'écran d'un événement).

---

## SPRINT 2 — Pipeline Big Data double chemin (label: epic-detection)

### Issue 2.1
**Titre:** [Sprint 2] 2.1 - Double chemin Kafka + Filebeat
**Corps:**
En tant que Système, je dois expédier les logs vers Kafka (edge_collector) ET vers Logstash (Filebeat), en parallèle, sans dépendance entre les deux chemins.

**Livrable attendu:** Topic Kafka network-features recevant des messages ET pipeline Filebeat->Logstash actif simultanément.

---

### Issue 2.2
**Titre:** [Sprint 2] 2.2 - Parsing et enrichissement GeoIP
**Corps:**
En tant que Système, je dois parser et enrichir les données (Logstash), notamment par résolution GeoIP.

**Livrable attendu:** Logstash pipeline avec filtre GeoIP fonctionnel, champs géo visibles dans les documents indexés.

---

### Issue 2.3
**Titre:** [Sprint 2] 2.3 - Indexation Elasticsearch (heap limité)
**Corps:**
En tant qu'Architecte, je veux indexer ces données dans Elasticsearch en limitant le heap JVM pour respecter le budget mémoire des VMs.

**Livrable attendu:** Cluster Elasticsearch single-node opérationnel, heap configuré (-Xms/-Xmx), index template créé.

---

### Issue 2.4
**Titre:** [Sprint 2] 2.4 - Dashboard Kibana
**Corps:**
En tant qu'Architecte, je veux construire un tableau de bord Kibana pour l'audit et la supervision humaine (non-critique).

**Livrable attendu:** Dashboard Kibana affichant le trafic en temps quasi-réel (capture d'écran).

---

## SPRINT 3 — Modèle LSTM-VAE (label: epic-detection)

### Issue 3.1
**Titre:** [Sprint 3] 3.1 - Intégration dataset CIC-IDS-2017
**Corps:**
En tant que Data Scientist, je dois intégrer et nettoyer le dataset réel CIC-IDS-2017 (trafic BENIGN) via pandas, avec lecture par chunks.

**Livrable attendu:** Script de chargement/nettoyage fonctionnel, dataset BENIGN filtré et prêt à l'entraînement.

---

### Issue 3.2
**Titre:** [Sprint 3] 3.2 - Architecture LSTM-VAE
**Corps:**
En tant qu'Architecte, je veux concevoir l'architecture LSTM-VAE (fenêtre glissante de 10 flux, espace latent dimension 2).

**Livrable attendu:** Modèle Keras 3 compilé, résumé d'architecture (model.summary()) documenté.

---

### Issue 3.3
**Titre:** [Sprint 3] 3.3 - Seuil dynamique μ+3σ
**Corps:**
En tant que Système, je dois calculer le seuil dynamique μ+3σ sur le jeu de validation et le sauvegarder (joblib) à côté du scaler.

**Livrable attendu:** lstm_candidate.h5 + global_scaler.pkl + seuil sauvegardés, courbes de convergence.

---

### Issue 3.4
**Titre:** [Sprint 3] 3.4 - API d'inférence temps réel (<15ms)
**Corps:**
En tant que Système, je dois exposer ce modèle via une API locale optimisée (chargement unique, warm-up, appel direct) pour une inférence <15ms.

**Livrable attendu:** Script réalisant une inférence unitaire mesurée <15ms, logs de latence à l'appui.

---

## SPRINT 4 — Orchestrateur SOAR (label: epic-prevention)

### Issue 4.1
**Titre:** [Sprint 4] 4.1 - Consolidation des alertes (SOAR)
**Corps:**
En tant que Moteur SOAR, je dois consolider les alertes déterministes (Suricata) et probabilistes (MSE) pour déclencher un playbook de sécurité.

**Livrable attendu:** Script SOAR recevant les deux types d'alertes et appliquant une logique de décision unifiée.

---

### Issue 4.2
**Titre:** [Sprint 4] 4.2 - Blocage iptables automatique
**Corps:**
En tant que Système, je dois interagir avec iptables pour isoler instantanément l'IP malveillante détectée.

**Livrable attendu:** Test hping3 bloqué en conditions réelles, latence mesurée et documentée.

---

### Issue 4.3
**Titre:** [Sprint 4] 4.3 - Vérification liste d'actifs critiques
**Corps:**
En tant que SOAR, je dois vérifier l'IP ciblée contre la liste d'actifs critiques avant toute exécution de blocage.

**Livrable attendu:** Blocage refusé automatiquement pour une IP whitelistée, log de refus généré.

---

## SPRINT 5 — Bridge Agent IA / Ollama (label: epic-prevention)

### Issue 5.1
**Titre:** [Sprint 5] 5.1 - Transmission asynchrone vers Agent IA
**Corps:**
En tant que SOAR, je dois transmettre un payload JSON (IP, MSE, métriques) à l'Agent IA de manière asynchrone, sans bloquer la décision de blocage.

**Livrable attendu:** Mécanisme asynchrone (thread/queue Kafka dédiée) démontré, blocage non ralenti.

---

### Issue 5.2
**Titre:** [Sprint 5] 5.2 - Diagnostic sémantique via Ollama
**Corps:**
En tant qu'Agent IA, je dois interroger l'API locale Ollama (Llama3, version quantifiée) pour interpréter sémantiquement l'anomalie.

**Livrable attendu:** Réponse LLM cohérente sur un cas de test (ex: SYN Flood correctement identifié).

---

### Issue 5.3
**Titre:** [Sprint 5] 5.3 - Validation stricte des règles générées
**Corps:**
En tant que Système, je dois valider strictement (regex, whitelist) toute règle générée par le LLM avant exécution.

**Livrable attendu:** Garde-fou testé avec une entrée malformée/hallucinée, rejet démontré.

---

### Issue 5.4
**Titre:** [Sprint 5] 5.4 - Génération de rapport d'incident SOC
**Corps:**
En tant qu'Agent IA, je dois rédiger et horodater un rapport d'incident SOC structuré.

**Livrable attendu:** Exemple de rapport SOC généré automatiquement à partir d'une vraie alerte.

---

## SPRINT 6 — Gouvernance & Human-in-the-Loop (label: epic-prevention)

### Issue 6.1
**Titre:** [Sprint 6] 6.1 - Liste d'actifs critiques
**Corps:**
En tant qu'Administrateur, je dois définir une liste d'actifs critiques ne pouvant jamais être bloqués automatiquement.

**Livrable attendu:** Fichier de configuration de la whitelist, chargé et utilisé par le SOAR.

---

### Issue 6.2
**Titre:** [Sprint 6] 6.2 - Workflow d'approbation avec timeout
**Corps:**
En tant que SOAR, je dois soumettre une demande d'approbation à l'admin avant toute action sur un actif critique, avec timeout configurable.

**Livrable attendu:** Démonstration complète du workflow (demande -> décision -> exécution).

---

### Issue 6.3
**Titre:** [Sprint 6] 6.3 - Log d'audit immuable
**Corps:**
En tant que Système, je dois journaliser chaque décision (automatique ou manuelle) dans un log d'audit immuable.

**Livrable attendu:** Log d'audit consultable, entrée générée pour chaque test précédent.

---

### Issue 6.4
**Titre:** [Sprint 6] 6.4 - Validation admin Champion/Challenger
**Corps:**
En tant qu'Administrateur, je dois valider ou rejeter un nouveau modèle (Champion/Challenger) via un rapport de performance avant mise en production.

**Livrable attendu:** Rapport de comparaison généré, décision admin simulée et appliquée.

---

## SPRINT 7 — Agent CTI (label: epic-anticipation)

### Issue 7.1
**Titre:** [Sprint 7] 7.1 - Collecte OSINT périodique
**Corps:**
En tant qu'Agent CTI, je dois collecter périodiquement l'actualité cyber depuis des sources fiables (CVE/NVD, CERT, advisories).

**Livrable attendu:** Script de collecte fonctionnel, exemples d'items collectés.

---

### Issue 7.2
**Titre:** [Sprint 7] 7.2 - Filtrage et classification de pertinence
**Corps:**
En tant qu'Agent CTI, je dois filtrer et classifier la pertinence d'une information (technologie concernée, criticité).

**Livrable attendu:** Exemple de classification correcte sur un jeu de test (ex: faille Next.js -> équipe dev).

---

### Issue 7.3
**Titre:** [Sprint 7] 7.3 - Alimentation Threat DB
**Corps:**
En tant que Système, je dois alimenter et enrichir la Threat DB locale à partir des informations validées.

**Livrable attendu:** Threat DB peuplée, requêtable, avec au moins quelques IOCs réels.

---

### Issue 7.4
**Titre:** [Sprint 7] 7.4 - Mise à jour des règles Suricata
**Corps:**
En tant qu'Agent CTI, je dois proposer une mise à jour des règles Suricata à partir des nouveaux IOCs.

**Livrable attendu:** Nouvelle règle Suricata générée et intégrée, testée.

---

## SPRINT 8 — Agent Formation (label: epic-anticipation)

### Issue 8.1
**Titre:** [Sprint 8] 8.1 - Identification du profil d'audience
**Corps:**
En tant qu'Agent Formation, je dois identifier le profil de l'audience concernée par une menace (global ou équipe technique spécifique).

**Livrable attendu:** Mapping profils/utilisateurs fonctionnel, testé sur un cas.

---

### Issue 8.2
**Titre:** [Sprint 8] 8.2 - Génération de contenu pédagogique via LLM
**Corps:**
En tant qu'Agent Formation, je dois générer un contenu pédagogique contextualisé via le LLM local, adapté à l'audience.

**Livrable attendu:** Deux exemples de contenus générés (profil global vs profil technique).

---

### Issue 8.3
**Titre:** [Sprint 8] 8.3 - Diffusion selon le déclencheur
**Corps:**
En tant que Système, je dois diffuser la formation par le canal approprié selon le déclencheur (proactif ou réactif).

**Livrable attendu:** Email/notification réellement envoyé et reçu, capture d'écran.

---

## SPRINT 9 — Industrialisation MLOps (label: epic-anticipation)

### Issue 9.1
**Titre:** [Sprint 9] 9.1 - Conteneurisation Docker (4 réseaux)
**Corps:**
En tant qu'Architecte, je dois conteneuriser chaque composant selon 4 réseaux Docker isolés (Edge, ML, Réponse, Veille).

**Livrable attendu:** docker-compose.yml complet, stack démarrable en une commande.

---

### Issue 9.2
**Titre:** [Sprint 9] 9.2 - Volume partagé pour le modèle
**Corps:**
En tant que Système, je dois partager le modèle entraîné (.h5/.pkl) entre conteneurs via un volume Docker nommé.

**Livrable attendu:** Volume nommé fonctionnel, modèle accessible en lecture par le conteneur d'inférence.

---

### Issue 9.3
**Titre:** [Sprint 9] 9.3 - Rapport de comparaison Champion/Challenger
**Corps:**
En tant que Pipeline MLOps, je dois générer automatiquement un rapport de comparaison Champion/Challenger avant tout remplacement de modèle.

**Livrable attendu:** Cycle complet démontré : nouveau modèle -> rapport -> validation -> promotion.

---

# Milestones à créer (un par Sprint)

Aller dans Issues -> Milestones -> New milestone, pour chacun :

- Sprint 1 — Déploiement Suricata
- Sprint 2 — Pipeline Big Data
- Sprint 3 — Modèle LSTM-VAE
- Sprint 4 — Orchestrateur SOAR
- Sprint 5 — Bridge Agent IA
- Sprint 6 — Gouvernance & Human-in-the-Loop
- Sprint 7 — Agent CTI
- Sprint 8 — Agent Formation
- Sprint 9 — Industrialisation MLOps

Assigner chaque issue créée ci-dessus au Milestone correspondant à son sprint.
