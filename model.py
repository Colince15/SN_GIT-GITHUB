import joblib
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import os

# Définir le chemin du modèle
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

def train_and_save_model():
    """
    Crée, entraîne et sauvegarde un modèle de classification factice.
    """
    print("Début de l'entraînement du modèle...")
    # Créer des données factices
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, random_state=42)

    # Créer et entraîner un modèle simple
    model = LogisticRegression(C=0.1) # Amélioration du modèle
    model.fit(X, y)
    print("Modèle entraîné.")

    # Créer le répertoire du modèle s'il n'existe pas
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Sauvegarder le modèle
    joblib.dump(model, MODEL_PATH)
    print(f"Modèle sauvegardé à l'emplacement : {MODEL_PATH}")

def predict(input_data):
    """
    Charge le modèle sauvegardé et simule une prédiction.
    
    Args:
        input_data (list or np.array): Les données d'entrée pour la prédiction.
        
    Returns:
        La prédiction du modèle.
    """
    print("Chargement du modèle pour la prédiction...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Le fichier du modèle n'a pas été trouvé à l'emplacement : {MODEL_PATH}. Veuillez d'abord entraîner le modèle.")
        
    model = joblib.load(MODEL_PATH)
    print("Modèle chargé.")
    
    prediction = model.predict(input_data)
    print(f"Prédiction : {prediction}")
    return prediction

if __name__ == "__main__":
    # Entraîner et sauvegarder le modèle
    train_and_save_model()
    
    # Exemple de prédiction
    # Créer des données de test factices (doivent avoir le même nombre de caractéristiques que les données d'entraînement)
    sample_input = [[0.1] * 10] 
    try:
        predict(sample_input)
    except FileNotFoundError as e:
        print(e)
