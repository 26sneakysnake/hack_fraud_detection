#!/usr/bin/env python3
"""
Script de test rapide pour explorer le projet de détection de fraude
"""
import json
import pandas as pd
from pathlib import Path

def main():
    print("=" * 70)
    print("🎯 TEST DU PROJET DE DÉTECTION DE FRAUDE")
    print("=" * 70)

    # 1. Vérifier le modèle
    print("\n📦 1. MODÈLE ENTRAÎNÉ")
    model_path = Path("models/fraud_detection_model.pkl")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ Modèle trouvé : {model_path}")
        print(f"   📏 Taille : {size_mb:.1f} MB")
    else:
        print(f"   ❌ Modèle non trouvé")
        return

    # 2. Charger et afficher les métriques
    print("\n📊 2. MÉTRIQUES DU MODÈLE")
    with open("results/training_results.json", 'r') as f:
        results = json.load(f)

    print(f"   🏆 Meilleur modèle : {results['best_model'].upper()}")
    print(f"   🎯 Seuil optimal : {results['optimal_threshold']:.4f}")
    print(f"\n   Métriques de performance :")
    for metric, value in results['metrics'].items():
        print(f"      • {metric:12s}: {value:.4f} ({value*100:.2f}%)" if value < 1 else f"      • {metric:12s}: {value:.4f}")

    # 3. Analyser les prédictions
    print("\n🔮 3. PRÉDICTIONS GÉNÉRÉES")
    submission = pd.read_csv("results/submission.csv")
    predictions_detailed = pd.read_csv("results/predictions_with_probabilities.csv")

    total = len(submission)
    frauds = submission['fraud_prediction'].sum()
    fraud_rate = frauds / total * 100

    print(f"   📝 Total de prédictions : {total:,}")
    print(f"   🚨 Fraudes détectées : {frauds:,} ({fraud_rate:.1f}%)")
    print(f"   ✅ Transactions légitimes : {total - frauds:,} ({100-fraud_rate:.1f}%)")

    # 4. Top features
    print("\n⭐ 4. TOP 5 FEATURES LES PLUS IMPORTANTES")
    features = pd.read_csv("results/feature_importance.csv")
    for idx, row in features.head(5).iterrows():
        print(f"   {idx+1}. {row['feature']:30s} → {row['importance']:.4f} ({row['importance']*100:.2f}%)")

    # 5. Distribution des probabilités
    print("\n📈 5. DISTRIBUTION DES PROBABILITÉS DE FRAUDE")
    proba = predictions_detailed['fraud_probability']
    print(f"   • Probabilité moyenne : {proba.mean():.4f}")
    print(f"   • Probabilité médiane : {proba.median():.4f}")
    print(f"   • Min / Max : {proba.min():.4f} / {proba.max():.4f}")

    high_risk = (proba > 0.8).sum()
    medium_risk = ((proba >= 0.3) & (proba <= 0.8)).sum()
    low_risk = (proba < 0.3).sum()

    print(f"\n   Répartition par niveau de risque :")
    print(f"   🔴 Risque élevé (>80%)   : {high_risk:,} ({high_risk/total*100:.1f}%)")
    print(f"   🟠 Risque moyen (30-80%) : {medium_risk:,} ({medium_risk/total*100:.1f}%)")
    print(f"   🟢 Risque faible (<30%)  : {low_risk:,} ({low_risk/total*100:.1f}%)")

    # 6. Visualisations disponibles
    print("\n🎨 6. VISUALISATIONS GÉNÉRÉES")
    figures_dir = Path("results/figures")
    if figures_dir.exists():
        figures = list(figures_dir.glob("*.png"))
        print(f"   📊 {len(figures)} graphiques disponibles dans results/figures/")
        for fig in figures[:5]:
            print(f"      • {fig.name}")
        if len(figures) > 5:
            print(f"      ... et {len(figures) - 5} autres")

    # 7. Commandes suivantes
    print("\n" + "=" * 70)
    print("🚀 PROCHAINES ÉTAPES")
    print("=" * 70)
    print("\n1️⃣  Lancer le dashboard interactif :")
    print("   streamlit run dashboard/app.py")
    print("\n2️⃣  Générer de nouvelles prédictions :")
    print("   python scripts/predict.py")
    print("\n3️⃣  Ré-entraîner le modèle :")
    print("   python scripts/train.py")
    print("\n4️⃣  Ouvrir les visualisations :")
    print("   start results\\figures  # Windows")
    print("\n5️⃣  Lire la documentation :")
    print("   • README.md - Vue d'ensemble")
    print("   • HACKATHON_GUIDE.md - Guide de présentation")
    print("   • GUIDE_DECOUVERTE.md - Guide de découverte")
    print("   • PROJECT_STATUS.md - État technique complet")

    print("\n" + "=" * 70)
    print("✅ TEST TERMINÉ AVEC SUCCÈS !")
    print("=" * 70)

if __name__ == "__main__":
    main()
