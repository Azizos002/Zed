# Documentation Exhaustive : Modèle Industriel LSTM-VAE pour NIDS Temps Réel

Ce document détaille l'architecture complète, la préparation des données, et les fondements mathématiques du modèle **LSTM-VAE** (Long Short-Term Memory - Variational Autoencoder) déployé pour la détection d'anomalies réseau en temps réel.

---

## 1. Philosophie de l'Architecture

Le système repose sur l'hybridation de deux concepts majeurs en intelligence artificielle :

1. **LSTM (Réseau de Neurones Récurrents) :** Capable de mémoriser des dépendances temporelles. Le trafic réseau n'est pas statique ; l'ordre dans lequel les flux arrivent est crucial.
2. **VAE (Autoencodeur Variationnel) :** Contrairement à un autoencodeur classique qui compresse en points fixes, le VAE compresse les données en **distributions probabilistes**. Cela permet au modèle de comprendre qu'un trafic normal a des variations naturelles, réduisant ainsi drastiquement les faux positifs en production.

---

## 2. Le Pipeline de Données (Data Engineering)

Avant d'entrer dans l'IA, les données brutes (`eve.json` via Suricata) subissent deux transformations mathématiques critiques.

### 2.1. Extraction des Features (Vecteur $X$)

Pour chaque flux TCP/UDP, nous extrayons un vecteur de 4 dimensions :

- `pkts_in` : Nombre de paquets vers le serveur.
- `pkts_out` : Nombre de paquets vers le client.
- `bytes_in` : Volume de données entrantes (octets).
- `bytes_out` : Volume de données sortantes (octets).

### 2.2. Normalisation Absolue (MinMaxScaler)

Les réseaux de neurones ne peuvent pas traiter simultanément des valeurs minuscules (1 paquet) et gigantesques (1 000 000 d'octets). Nous appliquons une mise à l'échelle stricte :

$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

**Contrainte Industrielle :** Les bornes sont "figées" sur des limites de production pour éviter que l'échelle ne change en plein vol.

- $X_{min} = [0, 0, 0, 0]$
- $X_{max} = [1500, 1500, 1000000, 1000000]$

### 2.3. Fenêtrage Temporel (Sliding Window)

Pour que le LSTM comprenne le contexte, les données ne sont pas analysées flux par flux, mais par blocs temporels successifs (Séquences).

- **Paramètre `TIME_STEPS = 10` :** Le modèle analyse le comportement du réseau sur des fenêtres de 10 flux consécutifs. 
- La forme tensorielle d'entrée passe donc de `(batch_size, 4)` à `(batch_size, 10, 4)`.

---

## 3. L'Architecture du Modèle (Couche par Couche)

L'architecture est construite en "sablier" (bottleneck) asymétrique via l'API fonctionnelle de Keras 3.

### Partie A : L'Encodeur (Compression Temporelle)

L'objectif de l'encodeur est de lire la séquence de 10 flux et d'en extraire l'essence mathématique.

1. **Input Layer :** Reçoit le tenseur de dimension `(10, 4)`.
2. **Couche LSTM (16 unités) :** * Traite la séquence temporelle.
  - L'activation est `tanh` (tangente hyperbolique) pour l'efficacité mathématique et éviter l'explosion du gradient.
  - `return_sequences=False` : Le LSTM lit les 10 flux temporels mais ne recrache qu'un seul vecteur résumé de dimension 16.
3. **Couches Dense (L'Espace Latent Probabiliste) :**
  - Au lieu d'avoir un vecteur fixe, le LSTM est scindé en deux réseaux denses parallèles pour définir une loi de probabilité (une distribution gaussienne).
  - `**z_mean` (Vecteur de Moyenne $\mu$) :** Définit le centre de la distribution.
  - `**z_log_var` (Vecteur de Log-Variance $\log(\sigma^2)$) :** Définit la largeur (l'incertitude) de la distribution.

### Partie B : Le Reparameterization Trick (Échantillonnage)

Pour que la rétropropagation du gradient (backpropagation) puisse fonctionner lors de l'entraînement, on ne peut pas intégrer une fonction purement aléatoire. Nous utilisons "l'astuce de reparamétrisation" :

$$z = \mu + \exp\left(\frac{\log(\sigma^2)}{2}\right) \cdot \epsilon$$

Où $\epsilon \sim \mathcal{N}(0, 1)$ est un bruit gaussien standard. Cette couche projette la séquence compressée dans un espace latent à 2 dimensions (`latent_dim = 2`).

### Partie C : Le Décodeur (Reconstruction Temporelle)

L'objectif du décodeur est de prendre le point $z$ (dimension 2) et de recréer parfaitement la séquence originale `(10, 4)`.

1. **Couche RepeatVector :** Prend le point compressé (dimension 2) et le duplique 10 fois pour restaurer la dimension temporelle `(10, 2)`.
2. **Couche LSTM (16 unités) :** * Lit la séquence dupliquée pour reconstituer la dynamique d'évolution temporelle.
  - `return_sequences=True` : Recrache un vecteur pour chaque pas de temps.
3. **Couche TimeDistributed(Dense) :** Applique une régression linéaire standard indépendamment sur chacun des 10 pas de temps pour retrouver les 4 dimensions originales (paquets et octets). Le tenseur final est de nouveau `(10, 4)`.

---

## 4. La Fonction de Perte (Custom VAELossLayer)

C'est le cœur mathématique du modèle. La perte totale minimisée par l'optimiseur **Adam** est la somme de deux pénalités distinctes :

$$\mathcal{L}*{Total} = \mathcal{L}*{Recon} + \mathcal{L}_{KL}$$

### 4.1. Reconstruction Loss (Erreur de Prédiction)

L'erreur quadratique moyenne (MSE) entre les données entrantes $X$ et les données reconstruites $\hat{X}$. Elle force le modèle à savoir décompresser l'information.

$$\mathcal{L}_{Recon} =  X - \hat{X} ^2$$

### 4.2. Kullback-Leibler (KL) Divergence (Régularisation)

Elle force les distributions latentes ($\mu$, $\sigma$) à se rapprocher d'une distribution normale centrée-réduite $\mathcal{N}(0, 1)$. Sans la perte KL, le modèle tricherait en éloignant les points à l'infini pour éviter qu'ils ne se chevauchent.

$$\mathcal{L}*{KL} = -\frac{1}{2} \sum*{i=1}^{k} \left( 1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2 \right)$$

---

## 5. Logique d'Inférence et Détection Zero-Day (Control Plane)

En production, le modèle utilise une logique **Non-Supervisée**. 

1. L'IA ne connaît **aucune** signature d'attaque (pas de malware pré-enregistré). Elle n'a été entraînée que sur la baseline saine du réseau de l'entreprise.
2. Lorsqu'un flux arrive, il est injecté dans le modèle.
3. **Cas Trafic Sain :** Le trafic ressemble à ce que l'IA a appris. Le décodeur reconstruit le flux facilement. La distance MSE entre l'entrée et la sortie est infime (ex: `0.02`). L'alerte est au vert.
4. **Cas Attaque (ex: SYN Flood, Déni de Service) :** La structure asymétrique des paquets (ex: 2 entrants, 0 sortant) sature les bornes du scaler. Le vecteur résultant atterrit dans une zone "vide" de l'espace latent. Le décodeur échoue totalement à le reconstruire. La MSE explose (ex: `15.50`).
5. Si $\text{MSE} > \text{SEUIL}$, l'infrastructure déclenche un événement critique dans Kafka.

