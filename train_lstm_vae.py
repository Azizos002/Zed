import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed, Lambda
from tensorflow.keras import backend as K

print("[*] Initialisation du Control Plane (Pipeline MLOps)...")

# ==========================================
# 1. EXTRACTION DES DONNÉES (Elasticsearch)
# ==========================================
# /!\ REMPLACE PAR L'IP DE TA MACHINE B (ELK) /!\
es = Elasticsearch(["http://192.168.56.130:9200"]) 

print("[*] Connexion à Elasticsearch et extraction des logs de flux...")
res = es.search(index="suricata-*", body={
    "query": {"match": {"event_type": "flow"}},
    "size": 5000,
    "sort": [{"@timestamp": {"order": "desc"}}]
})

# Nettoyage et création du DataFrame
data = []
for hit in res['hits']['hits']:
    flow = hit['_source'].get('flow', {})
    if flow:
        data.append([
            flow.get('pkts_toserver', 0),
            flow.get('pkts_toclient', 0),
            flow.get('bytes_toserver', 0),
            flow.get('bytes_toclient', 0)
        ])

df = pd.DataFrame(data, columns=['pkts_in', 'pkts_out', 'bytes_in', 'bytes_out'])
print(f"[+] Données extraites : {df.shape[0]} flux réseau.")

# ==========================================
# 2. FEATURE ENGINEERING (Normalisation)
# ==========================================
print("[*] Normalisation MinMaxScaler...")
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

time_steps = 10
num_features = df_scaled.shape[1]

def create_sequences(X, time_steps):
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
    return np.array(Xs)

X_train = create_sequences(df_scaled, time_steps)
print(f"[+] Tenseur d'entraînement créé : {X_train.shape} (Batch, TimeSteps, Features)")

# ==========================================
# 3. ARCHITECTURE LSTM-VAE (Le Vrai Modèle)
# ==========================================
print("[*] Construction du modèle LSTM-VAE...")
latent_dim = 4 # Dimension de l'espace latent probabiliste

# --- ENCODEUR ---
inputs = Input(shape=(time_steps, num_features), name='encoder_input')
h = LSTM(16, activation='relu', return_sequences=False)(inputs)

# L'espace latent (Moyenne et Log-Variance)
z_mean = Dense(latent_dim, name='z_mean')(h)
z_log_var = Dense(latent_dim, name='z_log_var')(h)

# Fonction d'échantillonnage (Reparameterization Trick: z = mean + sigma * epsilon)
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Couche Lambda pour injecter l'échantillonnage dans le réseau
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# --- DÉCODEUR ---
decoder_inputs = RepeatVector(time_steps)(z)
h_decoded = LSTM(16, activation='relu', return_sequences=True)(decoder_inputs)
outputs = TimeDistributed(Dense(num_features))(h_decoded)

# --- INSTANCIATION DU MODÈLE ---
vae = Model(inputs, outputs, name='lstm_vae')

# --- FONCTION DE PERTE CUSTOM (MSE + KL Divergence) ---
# 1. Erreur de reconstruction (MSE)
reconstruction_loss = tf.reduce_mean(tf.keras.losses.mse(inputs, outputs))
reconstruction_loss *= time_steps * num_features 

# 2. Divergence de Kullback-Leibler (Régularisation de l'espace latent)
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

# Perte totale
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

# ==========================================
# 4. ENTRAÎNEMENT & SAUVEGARDE
# ==========================================
print("[*] Démarrage de l'entraînement stochastique (Epochs=20)...")
# Note : Pas besoin de 'Y' avec add_loss(), X_train suffit comme cible
history = vae.fit(X_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

print("[+] Modèle entraîné. Sauvegarde en cours...")
vae.save("lstm_candidate.h5")

# ==========================================
# 5. GÉNÉRATION DE LA FIGURE
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Perte Totale d\'Entraînement (Loss)', color='blue')
plt.plot(history.history['val_loss'], label='Perte de Validation (Val Loss)', color='red')
plt.title('Convergence du Modèle LSTM-VAE - Détection d\'Anomalies')
plt.xlabel('Époques (Epochs)')
plt.ylabel('Loss (Reconstruction + KL Divergence)')
plt.legend()
plt.grid(True)
plt.savefig('figure_4_3_vae_loss.png')
print("[!] Graphique sauvegardé sous 'figure_4_3_vae_loss.png'.")