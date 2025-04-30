# pages/2_Classification.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import os
import tempfile # Nécessaire pour gérer le fichier modèle temporairement

# --- Fonctions de Chargement Modifiées ---

# @st.cache_resource # On cache la ressource, mais elle dépend du fichier uploadé
# Attention: Le cache fonctionne sur les arguments. Si l'utilisateur upload le *même* fichier,
# le cache sera utilisé. S'il upload un fichier différent (même nom), il rechargera.
def load_model_from_upload(uploaded_model_file):
    """Charge le modèle Keras à partir d'un fichier uploadé."""
    if uploaded_model_file is None:
        return None

    # Créer un fichier temporaire pour sauvegarder le modèle uploadé
    # TensorFlow a besoin d'un chemin de fichier pour charger le modèle .h5
    temp_model_path = None
    model = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            # Écrire le contenu du fichier uploadé dans le fichier temporaire
            tmp_file.write(uploaded_model_file.getvalue())
            temp_model_path = tmp_file.name # Obtenir le chemin du fichier temporaire

        # Charger le modèle depuis le chemin temporaire
        st.info(f"Chargement du modèle depuis le fichier temporaire: {temp_model_path}")
        model = tf.keras.models.load_model(temp_model_path, compile=False)
        st.success("Modèle chargé avec succès !")
        return model

    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle depuis le fichier uploadé : {e}")
        st.exception(e) # Affiche les détails de l'erreur pour le débogage
        return None
    finally:
        # Supprimer le fichier temporaire après le chargement (ou échec)
        if temp_model_path and os.path.exists(temp_model_path):
            try:
                os.remove(temp_model_path)
                # st.info(f"Fichier modèle temporaire supprimé: {temp_model_path}")
            except Exception as e:
                st.warning(f"Impossible de supprimer le fichier modèle temporaire {temp_model_path}: {e}")


# @st.cache_data # Cache basé sur le contenu du fichier uploadé
def load_labels_from_upload(uploaded_labels_file):
    """Charge les labels à partir d'un fichier .txt uploadé."""
    if uploaded_labels_file is None:
        return []
    try:
        # Lire le contenu directement depuis l'objet fichier uploadé
        stringio = uploaded_labels_file.getvalue().decode('utf-8')
        lines = stringio.splitlines()
        # Extraire les noms de classe
        class_names = [line.strip().split(' ', 1)[1] for line in lines if line.strip()]
        if not class_names:
            st.warning("Le fichier de labels semble vide ou mal formaté.")
            return []
        st.success(f"Labels chargés: {len(class_names)} classes trouvées.")
        return class_names
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier de labels uploadé : {e}")
        return []

# --- Fonctions de Prétraitement (inchangée) ---
def preprocess_image(image_pil, target_size=(224, 224)):
    """Prétraite l'image PIL pour correspondre à l'entrée du modèle."""
    image = ImageOps.fit(image_pil, target_size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)
    return data

# --- Interface Streamlit de la Page de Classification ---

st.title("🔬 Classification d'Images (Modèle Personnalisé)")
st.write("Chargez votre propre modèle Teachable Machine (.h5) et son fichier de labels (.txt), puis chargez une image à classifier.")
st.markdown("---") # Séparateur visuel

# --- Section pour Uploader le Modèle et les Labels ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Charger le Modèle")
    uploaded_model_file = st.file_uploader("Choisissez le fichier `keras_model.h5`", type=["h5"], key="model_uploader")

with col2:
    st.subheader("2. Charger les Labels")
    uploaded_labels_file = st.file_uploader("Choisissez le fichier `labels.txt`", type=["txt"], key="labels_uploader")

# --- Chargement et Logique principale ---
model = None
class_names = []

# Essayer de charger seulement si les fichiers sont présents
if uploaded_model_file is not None:
    # Utiliser une combinaison du nom et de la taille comme clé de cache simple
    # NOTE: Ceci n'est pas parfait, si le contenu change mais nom/taille restent, le cache peut être incorrect.
    # Une meilleure approche impliquerait de hacher le contenu, mais est plus complexe.
    @st.cache_resource(show_spinner="Chargement du modèle en cours...")
    def cached_load_model(file_id):
         # file_id n'est pas directement utilisé mais force le recalcul si l'upload change
        return load_model_from_upload(uploaded_model_file)
    # On utilise l'id interne de l'objet uploadé pour tenter de déclencher le re-calcul si le fichier change
    model = cached_load_model(uploaded_model_file.file_id)


if uploaded_labels_file is not None:
    @st.cache_data(show_spinner="Chargement des labels...")
    def cached_load_labels(file_id):
         return load_labels_from_upload(uploaded_labels_file)
    class_names = cached_load_labels(uploaded_labels_file.file_id)


# --- Section pour Uploader l'Image et Classifier (si modèle et labels chargés) ---
if model is not None and class_names:
    st.markdown("---")
    st.subheader("3. Charger une Image et Classifier")
    uploaded_image_file = st.file_uploader("Choisissez une image à classifier...", type=["jpg", "jpeg", "png"], key="image_classifier")

    if uploaded_image_file is not None:
        try:
            # Ouvrir et afficher l'image
            image = Image.open(uploaded_image_file).convert('RGB')
            st.image(image, caption="Image à classifier", use_column_width=True)
            st.write("")

            # Prétraiter et prédire
            with st.spinner("🧠 Classification en cours..."):
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                score = tf.nn.softmax(prediction[0]) # Appliquer Softmax pour probabilités
                predicted_index = np.argmax(score)
                predicted_class_name = class_names[predicted_index]
                confidence_score = np.max(score)

            # Afficher les résultats
            st.success(f"**Prédiction : {predicted_class_name}**")
            st.info(f"**Confiance : {confidence_score:.2%}**")

            with st.expander("Voir les probabilités détaillées"):
                results = {name: f"{prob:.2%}" for name, prob in zip(class_names, score)}
                st.write(results)

        except Exception as e:
            st.error(f"Une erreur est survenue lors de la classification de l'image : {e}")
            st.exception(e)
    else:
        st.info("Veuillez charger une image pour la classification.")

elif uploaded_model_file and uploaded_labels_file:
    # Ce cas se produit si le chargement a échoué mais que les fichiers ont été uploadés
    st.error("Le modèle ou les labels n'ont pas pu être chargés correctement. Vérifiez les messages d'erreur ci-dessus.")
else:
    st.info("Veuillez d'abord charger les fichiers `keras_model.h5` et `labels.txt`.")
