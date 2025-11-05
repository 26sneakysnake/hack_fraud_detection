# 🎯 CHECKLIST POUR OBTENIR 20/20 AU HACKATHON

## 🚀 Validation Automatique (Méthode Rapide)

```powershell
# 1. Activer l'environnement virtuel
.\venv\Scripts\activate

# 2. Lancer le script de validation
python validate_hackathon.py
```

✅ **Si tu obtiens 20/20, ton projet est prêt !**

---

## 📋 DÉMARCHE COMPLÈTE POUR S'ASSURER QUE TOUT MARCHE

### ✅ ÉTAPE 1 : Vérifier l'Installation

```powershell
# Tester que tous les packages sont installés
python -c "import numpy, pandas, sklearn, xgboost, lightgbm, streamlit; print('✅ Tous les packages OK')"
```

**Si erreur** : Refaire l'installation
```powershell
pip install -r requirements-windows.txt
```

---

### ✅ ÉTAPE 2 : Valider les Données Brutes

```powershell
# Vérifier que toutes les données sont présentes
python -c "
import os
from pathlib import Path

files = [
    'data/raw/transactions_train.csv',
    'data/raw/train_fraud_labels.json',
    'data/raw/cards_data.csv',
    'data/raw/users_data.csv',
    'data/raw/mcc_codes.json',
    'data/raw/evaluation_features.csv'
]

for f in files:
    if Path(f).exists():
        size = Path(f).stat().st_size / 1024
        print(f'✅ {f} ({size:.1f} KB)')
    else:
        print(f'❌ MANQUANT: {f}')
"
```

**Attendu** :
- ✅ Toutes les 6 fichiers présents
- ✅ Tailles raisonnables (transactions_train > 20 MB)

---

### ✅ ÉTAPE 3 : Valider le Modèle Entraîné

```powershell
# Vérifier que le modèle existe et se charge
python -c "
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

model_path = Path('models/fraud_detection_model.pkl')
if model_path.exists():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f'✅ Modèle chargé: {type(model).__name__}')
    print(f'✅ Taille: {size_mb:.1f} MB')
    if hasattr(model, 'n_features_in_'):
        print(f'✅ Features: {model.n_features_in_}')
else:
    print('❌ Modèle manquant. Lancer: python scripts/train.py')
"
```

**Attendu** :
- ✅ Modèle existe (5-10 MB)
- ✅ Type: RandomForestClassifier
- ✅ 38 features

**Si manquant** :
```powershell
python scripts/train.py  # Prend 5-10 minutes
```

---

### ✅ ÉTAPE 4 : Valider les Métriques du Modèle

```powershell
# Afficher et analyser les métriques
python -c "
import json

with open('results/training_results.json', 'r') as f:
    results = json.load(f)

print('🏆 MÉTRIQUES DU MODÈLE')
print('='*50)
print(f'Modèle: {results[\"best_model\"].upper()}')
print(f'Seuil optimal: {results[\"optimal_threshold\"]}')
print()

metrics = results['metrics']
print('Métriques de performance:')
for metric, value in metrics.items():
    status = '✅' if (
        (metric == 'pr_auc' and value >= 0.70) or
        (metric == 'roc_auc' and value >= 0.90) or
        (metric == 'f1' and value >= 0.60) or
        metric not in ['pr_auc', 'roc_auc', 'f1']
    ) else '⚠️'
    print(f'  {status} {metric:12s}: {value:.4f}')

print()
print('🎯 CRITÈRES TECHNIQUES:')
config = results['config']['model']
print(f'  ✅ Validation: {config[\"validation_strategy\"]} ({config[\"train_end_date\"]})')
print(f'  ✅ Rééquilibrage: {config[\"resampling_method\"].upper()}')
print(f'  ✅ Features: {len(results[\"features\"])}')
"
```

**Attendu** :
- ✅ PR-AUC ≥ 0.70 (métrique clé pour déséquilibre)
- ✅ ROC-AUC ≥ 0.90
- ✅ F1-Score ≥ 0.60
- ✅ Validation temporelle (2017-12-31)
- ✅ SMOTE appliqué
- ✅ 38+ features

---

### ✅ ÉTAPE 5 : Valider les Prédictions

```powershell
# Vérifier le fichier de soumission
python -c "
import pandas as pd

df = pd.read_csv('results/submission.csv')

print('📝 FICHIER DE SOUMISSION')
print('='*50)
print(f'Lignes: {len(df):,}')
print(f'Colonnes: {list(df.columns)}')
print()
print(f'Fraudes détectées: {df[\"fraud_prediction\"].sum():,}')
print(f'Taux de fraude: {df[\"fraud_prediction\"].mean()*100:.2f}%')
print()

# Vérifier la validité
if set(df.columns) != {'transaction_id', 'fraud_prediction'}:
    print('❌ COLONNES INCORRECTES')
elif not df['fraud_prediction'].isin([0, 1]).all():
    print('❌ VALEURS INVALIDES (doivent être 0 ou 1)')
elif len(df) != 90000:
    print(f'⚠️ ATTENTION: {len(df)} prédictions au lieu de 90,000')
else:
    print('✅ Format parfait !')
"
```

**Attendu** :
- ✅ 90,000 lignes exactement
- ✅ Colonnes: transaction_id, fraud_prediction
- ✅ Valeurs: 0 ou 1 uniquement
- ✅ ~55,000 fraudes détectées (60-65%)

**Si manquant ou incorrect** :
```powershell
python scripts/predict.py  # Prend 1-2 minutes
```

---

### ✅ ÉTAPE 6 : Tester le Dashboard

```powershell
# Lancer le dashboard
streamlit run dashboard/app.py
```

**Vérifications manuelles dans le dashboard** :
- ✅ Page se charge sans erreur
- ✅ Métriques affichées correctement
- ✅ Graphiques visibles (ROC, PR, confusion matrix, etc.)
- ✅ Top features affichées
- ✅ Navigation fluide

**Arrêter** : `Ctrl+C` dans le terminal

---

### ✅ ÉTAPE 7 : Vérifier les Visualisations

```powershell
# Lister les graphiques générés
python -c "
from pathlib import Path
figures = list(Path('results/figures').glob('*.png'))
print(f'📊 {len(figures)} VISUALISATIONS GÉNÉRÉES:')
print('='*50)
for i, fig in enumerate(figures, 1):
    size_kb = fig.stat().st_size / 1024
    print(f'  {i}. {fig.name} ({size_kb:.1f} KB)')
"
```

**Attendu** :
- ✅ Au moins 4 graphiques
- ✅ Fichiers PNG valides (> 10 KB chacun)

**Ouvrir les graphiques** :
```powershell
start results\figures  # Windows
```

---

### ✅ ÉTAPE 8 : Valider la Documentation

```powershell
# Vérifier la complétude de la documentation
python -c "
from pathlib import Path

docs = {
    'README.md': 5000,
    'PROJECT_STATUS.md': 5000,
    'HACKATHON_GUIDE.md': 10000,
    'GUIDE_DECOUVERTE.md': 5000
}

print('📚 DOCUMENTATION:')
print('='*50)
for doc, min_size in docs.items():
    if Path(doc).exists():
        size = Path(doc).stat().st_size
        status = '✅' if size >= min_size else '⚠️'
        print(f'{status} {doc:30s} {size:,} caractères')
    else:
        print(f'❌ {doc:30s} MANQUANT')
"
```

**Attendu** :
- ✅ README.md (5,000+ caractères)
- ✅ PROJECT_STATUS.md (5,000+ caractères)
- ✅ HACKATHON_GUIDE.md (10,000+ caractères)
- ✅ GUIDE_DECOUVERTE.md (5,000+ caractères)

---

### ✅ ÉTAPE 9 : Test d'Intégration Complet

```powershell
# Exécuter le test complet du projet
python test_project.py
```

**Attendu** :
- ✅ Toutes les sections s'affichent
- ✅ Aucune erreur
- ✅ Métriques cohérentes

---

### ✅ ÉTAPE 10 : Validation Finale avec Score

```powershell
# VALIDATION OFFICIELLE POUR LE HACKATHON
python validate_hackathon.py
```

**Objectif** : **20/20** 🏆

**Décomposition des 20 points** :

| Critère | Points | Validation |
|---------|--------|------------|
| **1. Livrables obligatoires** | 5 | submission.csv, modèle, résultats, visualisations |
| **2. Qualité du modèle** | 6 | PR-AUC ≥0.7, ROC-AUC ≥0.9, F1 ≥0.6, SMOTE, validation temporelle |
| **3. Feature engineering** | 3 | 30+ features, importance analysée, diversité |
| **4. Code et architecture** | 2 | Structure modulaire, reproductibilité |
| **5. Documentation** | 2 | README complet, docs techniques |
| **6. Présentation & Dashboard** | 2 | Dashboard fonctionnel, notebooks |

---

## 🎯 CRITÈRES DE RÉUSSITE DÉTAILLÉS

### ✅ Pour avoir 20/20, il FAUT :

#### 📦 **Livrables** (5 points)
- ✅ `results/submission.csv` avec 90,000 prédictions au format exact
- ✅ `models/fraud_detection_model.pkl` (5-10 MB)
- ✅ `results/training_results.json` avec toutes les métriques
- ✅ `results/figures/*.png` (4+ graphiques)

#### 🤖 **Qualité du modèle** (6 points)
- ✅ **PR-AUC ≥ 0.70** (CRITIQUE pour données déséquilibrées)
- ✅ **ROC-AUC ≥ 0.90**
- ✅ **F1-Score ≥ 0.60**
- ✅ SMOTE utilisé pour rééquilibrer les classes
- ✅ Validation temporelle (train: 2016-2017, val: 2018)

#### ⚙️ **Feature engineering** (3 points)
- ✅ Au moins 30 features créées
- ✅ `results/feature_importance.csv` généré
- ✅ Diversité : temporelles + agrégations + encodages

#### 💻 **Code** (2 points)
- ✅ Structure modulaire (`src/`, `scripts/`)
- ✅ `requirements.txt` ou `requirements-windows.txt`
- ✅ `scripts/train.py` et `scripts/predict.py` fonctionnels

#### 📚 **Documentation** (2 points)
- ✅ README.md > 5,000 caractères
- ✅ PROJECT_STATUS.md ou HACKATHON_GUIDE.md présent

#### 🎨 **Présentation** (2 points)
- ✅ Dashboard Streamlit fonctionnel
- ✅ Au moins 1 notebook d'analyse

---

## 🚨 ERREURS COURANTES À ÉVITER

### ❌ **Erreur 1 : Mauvais format de submission.csv**
```python
# ❌ MAUVAIS
transaction_id, prediction, probability
1, "fraud", 0.95

# ✅ BON
transaction_id, fraud_prediction
1, 1
```

### ❌ **Erreur 2 : Métriques insuffisantes**
- PR-AUC < 0.70 → Améliorer le feature engineering
- ROC-AUC < 0.90 → Tester d'autres modèles ou hyperparamètres
- F1 < 0.60 → Optimiser le seuil de décision

### ❌ **Erreur 3 : Pas de gestion du déséquilibre**
- ❌ Entraîner directement sur données déséquilibrées
- ✅ Utiliser SMOTE ou class_weight

### ❌ **Erreur 4 : Data leakage**
- ❌ Split aléatoire sur données temporelles
- ✅ Split temporel (train avant validation dans le temps)

### ❌ **Erreur 5 : Documentation insuffisante**
- ❌ README vide ou trop court
- ✅ README détaillé avec architecture, résultats, usage

---

## 🔧 SI TU N'AS PAS 20/20

### 🔴 Score < 16/20 : CRITIQUE
1. Vérifier que le modèle est entraîné : `python scripts/train.py`
2. Vérifier que les prédictions sont générées : `python scripts/predict.py`
3. Relancer `python validate_hackathon.py`

### 🟡 Score 16-19/20 : Amélioration mineure
- Consulter les messages du script de validation
- Corriger les points manquants indiqués
- Relancer `python validate_hackathon.py`

### 🟢 Score 20/20 : PARFAIT !
Tu es prêt pour le hackathon ! 🎉

---

## ⭐ BONUS : Dépasser 20/20

Pour obtenir des **points bonus** :

1. **🎓 Certification IBM watsonx** (+1 point)
   - Ajouter certificat dans `docs/certifications/`

2. **🧪 Tests unitaires** (+1 point)
   - Créer `tests/test_*.py`

3. **🔬 Comparaison de 3+ modèles** (+1 point)
   - Déjà fait ! (Random Forest, XGBoost, LightGBM, etc.)

4. **🔍 Analyse des erreurs** (+1 point)
   - Créer `results/error_analysis.csv`

5. **💡 Explainabilité (SHAP)** (+1 point)
   - Ajouter graphiques SHAP dans `results/figures/`

---

## 📞 AIDE & RESSOURCES

### 📖 Guides disponibles
- **GUIDE_DECOUVERTE.md** : Découvrir le projet pas à pas
- **HACKATHON_GUIDE.md** : Préparer ta présentation
- **PROJECT_STATUS.md** : État technique complet
- **README.md** : Instructions générales

### 🛠️ Scripts utiles
```powershell
python test_project.py          # Test rapide
python validate_hackathon.py    # Validation officielle
streamlit run dashboard/app.py  # Dashboard
python scripts/train.py         # Réentraîner
python scripts/predict.py       # Régénérer prédictions
```

---

## ✅ CHECKLIST FINALE AVANT SOUMISSION

Avant de soumettre ton projet, vérifie cette checklist :

- [ ] ✅ `python validate_hackathon.py` → **20/20**
- [ ] ✅ Dashboard fonctionne : `streamlit run dashboard/app.py`
- [ ] ✅ `results/submission.csv` existe et est valide
- [ ] ✅ Documentation complète (4 fichiers .md)
- [ ] ✅ Tout poussé sur GitHub
- [ ] ✅ Branche principale définie correctement
- [ ] ✅ README à jour avec instructions
- [ ] ✅ Pitch préparé (consulter HACKATHON_GUIDE.md)

---

## 🏆 TU ES PRÊT !

Si tu as **20/20** et que tous les points de la checklist finale sont validés, **ton projet est prêt pour le hackathon** !

Bonne chance ! 🚀🎯
