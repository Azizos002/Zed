# 🧠 Documentation : Modèle Prédictif LSTM-VAE (Control Plane)

## 📌 Description du Projet
Ce module constitue le **Control Plane (Cerveau)** de la plateforme de cyberdéfense autonome. 
Il implémente un Autoencodeur Variationnel basé sur des réseaux de neurones récurrents (LSTM-VAE). Son objectif est d'ingérer des séquences temporelles de trafic réseau (depuis Elasticsearch), d'apprendre la distribution probabiliste du trafic légitime, et de détecter les menaces furtives (Zero-Day, APT) via le calcul de l'erreur de reconstruction (MSE).

---

## 🏗️ Architecture du Modèle et Justification des Couches (Layers)

Le modèle est construit en 3 grands blocs via l'API fonctionnelle de Keras/TensorFlow. Le tenseur d'entrée est de dimension `(Batch_Size, Time_Steps, Features)`.

### 1. L'Encodeur (Compression Temporelle)
L'objectif de l'encodeur est de lire la séquence réseau et d'en extraire le contexte temporel.

* **`Input(shape=(time_steps, num_features))`** : 
    * *Rôle :* Définit la porte d'entrée du réseau.
    * *Justification :* Contrairement aux modèles classiques qui lisent ligne par ligne, nous injectons des "fenêtres glissantes" (ex: 10 paquets à la suite) pour que le modèle comprenne la notion de *comportement* dans le temps.
* **`LSTM(16, return_sequences=False)`** : 
    * *Rôle :* Apprendre les relations de causalité.
    * *Justification :* Les cellules LSTM (Long Short-Term Memory) conservent une mémoire à long terme grâce à leurs portes logiques. `return_sequences=False` signifie qu'après avoir lu les 10 pas de temps, le LSTM ne recrache qu'un seul vecteur global qui résume toute l'action.

### 2. L'Espace Latent (Le Cœur Probabiliste - VAE)
C'est ici que le VAE se distingue d'un Autoencodeur classique. Au lieu de compresser la séquence en un point rigide, il la transforme en une zone de probabilité.

* **`Dense(latent_dim, name='z_mean')` & `Dense(latent_dim, name='z_log_var')`** :
    * *Rôle :* Calculer l'espérance (moyenne) et la variance de notre distribution.
    * *Justification :* Cela permet de créer un modèle souple. Le trafic réseau fluctue naturellement ; ces couches permettent au modèle de tolérer le "bruit" légitime sans générer de faux positifs.
* **`Lambda(sampling)` (L'astuce de Reparamétrisation)** :
    * *Rôle :* Échantillonner un point $z$ dans l'espace latent.
    * *Justification :* En Deep Learning, on ne peut pas rétropropager (Backpropagation) à travers un nœud purement aléatoire. L'astuce $z = \mu + \sigma \odot \epsilon$ (où $\epsilon$ est un bruit gaussien) permet de garder l'aléatoire tout en rendant le calcul des dérivées possible pour mettre à jour les poids du réseau.

### 3. Le Décodeur (Reconstruction Séquentielle)
L'objectif est de prendre le point latent $z$ (qui n'a plus de notion de temps) et de le déplier pour recréer la séquence réseau d'origine.

* **`RepeatVector(time_steps)`** :
    * *Rôle :* Dupliquer le vecteur latent $z$.
    * *Justification :* Le décodeur LSTM a besoin d'une entrée 3D (avec du temps). Cette couche prend le point compressé et le "copie" 10 fois pour que le LSTM puisse commencer à reconstruire chaque étape temporelle.
* **`LSTM(16, return_sequences=True)`** :
    * *Rôle :* Recréer la chronologie.
    * *Justification :* Ici, `return_sequences=True` est crucial. Le LSTM doit recracher une prédiction pour *chaque* pas de temps (les 10 paquets originaux) et non plus un résumé global.
* **`TimeDistributed(Dense(num_features))`** :
    * *Rôle :* Prédire les caractéristiques finales (octets, etc.) pour chaque pas de temps.
    * *Justification :* Cette couche applique un réseau dense (Perceptron) indépendamment sur chaque instant $t$ de la séquence générée par le LSTM, garantissant que la dimension de sortie correspond exactement à la dimension d'entrée.

---

## 🧮 Fonction de Perte Personnalisée (Custom Loss)

Le modèle n'utilise pas une fonction de perte standard de Keras, car un VAE nécessite un équilibre entre deux forces mathématiques :

1.  **Reconstruction Loss (MSE - Mean Squared Error) :** * Pénalise le modèle si la séquence de sortie ne ressemble pas à la séquence d'entrée.
2.  **KL Divergence (Kullback-Leibler) :** * Agit comme un régularisateur. Elle force les distributions générées par `z_mean` et `z_log_var` à se rapprocher d'une loi normale centrée réduite $\mathcal{N}(0, 1)$. Sans cela, l'espace latent serait discontinu et le modèle ne pourrait pas repérer les attaques furtives efficacement.

---

## 🚀 Exécution (Pipeline MLOps)

### Prérequis
```bash
pip install elasticsearch pandas numpy scikit-learn tensorflow matplotlib