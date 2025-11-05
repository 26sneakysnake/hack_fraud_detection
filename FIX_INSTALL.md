# 🔧 RÉSOLUTION RAPIDE - Erreur d'installation Windows

## ⚠️ Votre problème

**Erreur :** `ModuleNotFoundError: No module named 'distutils'`

**Cause :** Python 3.13 + anciennes versions de numpy incompatibles

---

## ✅ SOLUTION EN 3 ÉTAPES

### Étape 1 : Nettoyer et recréer le venv

Ouvrez PowerShell dans le dossier `hack_fraud_detection` :

```powershell
# Désactiver le venv si activé
deactivate

# Supprimer l'ancien venv
Remove-Item -Recurse -Force venv

# Créer nouveau venv
python -m venv venv

# Activer
.\venv\Scripts\activate
```

### Étape 2 : Installer avec les nouvelles versions

```powershell
# Mettre à jour pip
python -m pip install --upgrade pip

# Installer avec requirements-windows.txt
pip install -r requirements-windows.txt
```

### Étape 3 : Tester

```powershell
# Vérifier l'installation
python -c "import numpy, pandas, sklearn; print('✅ Installation réussie!')"

# Tester les prédictions
python scripts/predict.py
```

---

## 🚀 UTILISATION

### Générer des prédictions :
```powershell
python scripts/predict.py
```

### Lancer le dashboard :
```powershell
streamlit run dashboard/app.py
```

### Voir les résultats :
```powershell
# Voir les prédictions
Get-Content results\submission.csv | Select-Object -First 20

# Voir les métriques
python -c "import json; print(json.dumps(json.load(open('results/training_results.json')), indent=2))"
```

---

## 💡 ALTERNATIVE : Installation package par package

Si `requirements-windows.txt` ne marche pas :

```powershell
pip install numpy pandas scikit-learn
pip install xgboost lightgbm
pip install matplotlib seaborn plotly
pip install imbalanced-learn tqdm
pip install streamlit joblib pyyaml
```

---

## 📊 DÉJÀ PRÊT À UTILISER

Même si l'installation échoue, vous avez déjà les fichiers générés :

✅ **results/submission.csv** - 90,000 prédictions
✅ **models/fraud_detection_model.pkl** - Modèle entraîné
✅ **results/figures/** - 4 visualisations PNG
✅ **results/training_results.json** - Toutes les métriques

Vous pouvez les utiliser directement sans réinstaller !

---

## ❓ BESOIN D'AIDE ?

Si le problème persiste :
1. Vérifier version Python : `python --version`
2. Si Python 3.13, considérer installer Python 3.11
3. Télécharger Python 3.11 : https://www.python.org/downloads/
