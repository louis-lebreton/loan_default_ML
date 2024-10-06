"""
Projet Scoring 
M2 MoSEF
Louis LEBRETON
Dataset: hmeq.csv

Script principal
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from xgboost import XGBClassifier

import functions_EDA as f_eda
import functions_pretraitement as f_pre
import functions_modelisation as f_m
import functions_tests_statistiques as f_ts


# EDA #######################################################################################

df = pd.read_csv("data/hmeq.csv")

df.head()
df.shape
df.info()
df.describe()

vars_cat = df.select_dtypes(include=['object', 'category']).columns
print(vars_cat)

vars_num = df.select_dtypes(include=['float', 'int']).columns
vars_num = list(vars_num)
vars_num.remove('BAD')
print(vars_num)

# Univarie

f_eda.afficher_df_head(df)
f_eda.afficher_cat_vars_univarie_graph(df, 'BAD', palette=["purple"])
f_eda.afficher_cat_vars_univarie_tableau(df, 'BAD')
f_eda.afficher_num_vars_univarie_graph(df, palette=["#ff1717"])


# Bivarie

ma_palette = ["#ff0000", "#3511ca"]

f_eda.afficher_cat_vars_bivarie_graph(df, 'BAD', palette = ma_palette)
f_eda.afficher_num_vars_bivarie_graph(df, 'BAD', palette = ma_palette)
f_eda.afficher_cat_vars_bivarie_tableau(df, 'BAD')

# correlation entre variables categorielles
f_eda.v_cramer(df['REASON'], df['JOB'])
f_eda.v_cramer(df['BAD'], df['JOB']) 
f_eda.v_cramer(df['BAD'], df['JOB'])

# correlation entre 1 var cat x 1 var num
for var in vars_num:
    print('correlation_ratio entre ',var ,' et BAD: ', f_eda.correlation_ratio(df['BAD'], df[var]))

# Multivarie

f_eda.afficher_matrice_correlation(df)
# Derog et Delag corrélés à la variable y
# Value et Mortdue fortement et postiviement corrélés
sns.regplot(x=df['VALUE'], y=df['MORTDUE'], data=df)
plt.xlabel('VALUE')
plt.ylabel('MORTDUE')
plt.title('VALUE x MORTDUE')
plt.show()

# Prétraitement des données #######################################################################################

# Traitement des valeurs manquantes

f_pre.afficher_pourcentage_valeurs_manquantes(df)

# Supression des lignes avec un nb de valeurs manquantes > 5
tab_nb_va_manquantes = f_pre.afficher_nb_valeurs_manquantes(df)
index_a_enlever = tab_nb_va_manquantes[tab_nb_va_manquantes > 5].index
df = df.drop(index=index_a_enlever)
df = df.reset_index(drop=True)

# Supression des vars avec un nb de valeurs manquantes > 15 %
df.drop(columns=['DEBTINC'], inplace=True)

# Imputation
df["REASON"].fillna("Other", inplace=True)
df["JOB"].fillna(df["JOB"].mode()[0],inplace=True)

# imputation par la moyenne
for var in ["MORTDUE", "YOJ", "NINQ", "CLAGE", "CLNO", "VALUE"]:
    df[var] = df[var].fillna(value=df[var].mean())

# imputation par 0
for var in ["DEROG", "DELINQ"]:
    df[var] = df[var].fillna(value=0)

f_pre.afficher_pourcentage_valeurs_manquantes(df)

# Encodage

df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded.columns

# Normalisation / Standardisation

df_normalise = f_pre.normaliser_variables(df_encoded, 'BAD')
f_eda.afficher_df_head(df_normalise)

df_clean = df_normalise.applymap(lambda x: int(x) if isinstance(x, bool) else x)
f_eda.afficher_df_head(df_clean)
df_clean.info()

# Feature Engineering

"""df["PROBINC"] = df.MORTDUE/df.DEBTINC # adding new feature, (current debt on mortgage)/(debt to income ratio)
"""

colonnes_pour_interaction = ["DEROG", "DELINQ"]
df_modelisation = f_pre.creation_variables_interaction(df_clean, cols = colonnes_pour_interaction)
df_modelisation.info()


# Modélisation #######################################################################################

# Choix des variables
vars_selectionne = f_m.stepwise_selection(df_modelisation, 'BAD')

# Test statistiques
f_ts.test_homoscedasticite(df.drop(columns=['BAD']), df['BAD'])
dw_stat = f_ts.test_independance_erreurs(df.drop(columns=['BAD']), df['BAD'])
vif_df= f_ts.test_independance_variables(df.drop(columns=['BAD']))

# Fine-tuning d'une reg log
best_model, best_params = f_m.regression_logistique_kfold_gridsearch(df_modelisation, 'BAD', vars_selectionne)

# Choix du meilleur modèle
resultats_test_modeles = f_m.tester_modeles(df_modelisation, target_variable = 'BAD')
