"""
Projet Scoring 
M2 MoSEF
Louis LEBRETON
Dataset: hmeq.csv

Fonctions pour les tests statistiques
"""
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


# Test homoscedasticite

def test_homoscedasticite(X, y):

    # modele logistique
    log_modele = LogisticRegression(max_iter=1000)
    log_modele.fit(X, y)
    
    # probas predites
    y_pred_prob = log_modele.predict_proba(X)[:, 1]

    # résidus de Pearson
    residus_pearson = (y - y_pred_prob) / np.sqrt(y_pred_prob * (1 - y_pred_prob))
    
    # graph des résidus vs probabilités prédites
    plt.scatter(y_pred_prob, residus_pearson, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('probabilités prédites')
    plt.ylabel('résidus de Pearson')
    plt.title('résidus de pearson vs probabilités prédites')
    plt.show()


# Test Indépendance des erreurs

def test_independance_erreurs(X, y):
    # modele logistique
    log_modele = LogisticRegression(max_iter=1000)
    log_modele.fit(X, y)
    
    # probas predites
    y_pred_prob = log_modele.predict_proba(X)[:, 1]
    
    # residus
    residuals = y - y_pred_prob
    
    # stats de durbin watson
    dw_stat = durbin_watson(residuals)
    
    return dw_stat


# Test indépendance des variables

def test_independance_variables(X):
    # ajout constante
    X = sm.add_constant(X)
    # calcul du VIF
    vif_data = pd.DataFrame()
    vif_data["variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    return vif_data
