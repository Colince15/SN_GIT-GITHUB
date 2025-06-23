# Mini-projet IA / GitHub / Hugging Face

Ce projet est une démonstration d'un flux de travail CI/CD pour un modèle d'intelligence artificielle factice. Il utilise GitHub pour le contrôle de version, GitHub Actions pour l'intégration continue et le déploiement continu, et Hugging Face Hub pour héberger le modèle entraîné.

## Description

Le script `model.py` contient les fonctions pour :
- Entraîner un modèle de classification simple (factice).
- Sauvegarder le modèle entraîné dans un fichier.
- Charger le modèle et effectuer une prédiction.

Le workflow GitHub Actions automatise l'entraînement et le déploiement du modèle sur Hugging Face à chaque push sur la branche `main`.
