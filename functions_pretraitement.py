"""
Projet Scoring 
M2 MoSEF
Louis LEBRETON
Dataset: hmeq.csv

Fonctions pour le prétraitement des données
"""
from itertools import combinations_with_replacement

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Traitement des valeurs manquantes

def afficher_pourcentage_valeurs_manquantes(df):
    pourcentage_manquantes = df.isna().mean() * 100
    pourcentage_manquantes = pourcentage_manquantes.sort_values(ascending=False)
    print("Pourcentage de valeurs manquantes par variable (en %) :\n")
    print(round(pourcentage_manquantes), 2)

def afficher_nb_valeurs_manquantes(df):
    lignes_avec_n_manquantes = df.isna().sum(axis=1) # nb de valeurs manquantes par ligne
    return lignes_avec_n_manquantes

# Normalisation

def normaliser_variables(df, y, scaler = "standard" ):

    vars_num = df.select_dtypes(include=['float', 'int']).columns
    vars_num = list(vars_num)
    vars_num.remove(y)

    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    elif scaler == 'robust':
        scaler = RobustScaler()

    df[vars_num] = scaler.fit_transform(df[vars_num])
    
    return df

def creation_variables_interaction(df, cols, polynomial = True):
    """
    Cette fonction permet de créer des variables d'intéraction à partir de colonnes données
    d'un df d'entrée
    """

    # boucle sur toutes les combinaisons avec remplacement
    for col1, col2 in combinations_with_replacement(cols, 2):
        
        if col1 == col2:
            if polynomial:
                df[f'{col1}**2'] = df[col1] ** 2
        else:
            # Si les deux colonnes sont différentes -> interaction
            df[f'{col1}*{col2}'] = df[col1] * df[col2]
    
    return df