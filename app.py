import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps # Pillow pour manipuler les images
import cv2 # OpenCV pour certains prétraitements si nécessaire

# --- Configuration et Chargement du Modèle ---

# Utiliser le cache de Streamlit pour ne charger le modèle qu'une seule fois
@st.cache_resource
def load_model():
    """Charge le modèle Keras depuis le fichier .h5"""
    try:
        model = tf.keras.models.load_model("https://github.com/ClementDetee/model/raw/refs/heads/main/keras_model.h5", compile=False) # compile=False est souvent nécessaire pour les modèles TM
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

@st.cache_data
def load_labels():
    """Charge les noms des classes depuis labels.txt"""
    try:
        with open("https://github.com/ClementDetee/model/raw/refs/heads/main/labels.txt", 'r') as f:
            # Lire les lignes, enlever les numéros et les espaces superflus
            class_names = [line.strip().split(' ', 1)[1] for line in f if line.strip()]
        return class_names
    except FileNotFoundError:
        st.error("Erreur : Le fichier 'labels.txt' est introuvable.")
        return []
    except Exception as e:
        st.error(f"Erreur lors de la lecture de labels.txt : {e}")
        return []

model = load_model()
class_names = load_labels()

# --- Fonctions de Prétraitement ---

def preprocess_image(image_pil, target_size=(224, 224)):
    """
    Prétraite l'image pour correspondre à l'entrée attendue par le modèle Teachable Machine.
    Args:
        image_pil (PIL.Image): L'image chargée avec Pillow.
        target_size (tuple): La taille attendue par le modèle (souvent 224x224 pour Teachable Machine).
    Returns:
        np.ndarray: L'image prétraitée sous forme de tableau NumPy.
    """
    # Redimensionner en gardant le ratio, puis 'padder' pour atteindre la taille cible
    # C'est la méthode que Teachable Machine utilise souvent dans ses extraits de code
    image = ImageOps.fit(image_pil, target_size, Image.Resampling.LANCZOS)

    # Convertir l'image en tableau NumPy
    image_array = np.asarray(image)

    # Convertir en RGB (si l'image est en RGBA ou autre)
    if image_array.shape[-1] == 4: # Gérer la transparence (canal Alpha)
       image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    elif len(image_array.shape) == 2: # Gérer les images en niveaux de gris
       image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    # Normaliser l'image (important ! Les modèles TM Keras attendent souvent des valeurs entre -1 et 1)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Ajouter une dimension de 'batch' (le modèle attend un lot d'images, même si c'est une seule)
    # La forme attendue est (1, height, width, channels)
    data = np.expand_dims(normalized_image_array, axis=0)

    return data

# --- Interface Streamlit ---

st.title("Classifier d'Images avec Teachable Machine")
st.write("Téléchargez une image et le modèle prédira sa classe.")

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if model is None or not class_names:
    st.warning("Le modèle ou les étiquettes n'ont pas pu être chargés. Vérifiez les fichiers et les messages d'erreur.")
else:
    if uploaded_file is not None:
        # Ouvrir l'image téléchargée avec Pillow
        try:
            image = Image.open(uploaded_file).convert('RGB') # S'assurer qu'elle est en RGB

            # Afficher l'image téléchargée
            st.image(image, caption="Image téléchargée", use_column_width=True)
            st.write("") # Espace
            st.write("Classification en cours...")

            # Prétraiter l'image
            processed_image = preprocess_image(image)

            # Faire la prédiction
            prediction = model.predict(processed_image)
            predicted_index = np.argmax(prediction) # Obtenir l'index de la classe avec la plus haute probabilité
            predicted_class_name = class_names[predicted_index]
            confidence_score = prediction[0][predicted_index] # Obtenir le score de confiance

            # Afficher les résultats
            st.success(f"Prédiction : **{predicted_class_name}**")
            st.info(f"Confiance : **{confidence_score:.2%}**")

            # Optionnel : Afficher toutes les probabilités
            # st.write("Probabilités par classe :")
            # results = {name: f"{prob:.2%}" for name, prob in zip(class_names, prediction[0])}
            # st.write(results)

        except Exception as e:
            st.error(f"Une erreur est survenue lors du traitement de l'image : {e}")
