"""
Projet Scoring 
M2 MoSEF
Louis LEBRETON
Dataset: hmeq.csv

main.py: Script principal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
print(vars_cat) #vars cat sans 'BAD'

for var in vars_cat:
    print(df[var].nunique(),' modalités issues de la variable', str(var) + " : ", df[var].unique())

vars_num = df.select_dtypes(include=['float', 'int']).columns
vars_num = list(vars_num)
vars_num.remove('BAD')
print(vars_num)

# Univarie

f_eda.afficher_df_head(df)
f_eda.afficher_cat_vars_univarie_graph(df, 'BAD', palette=["purple"])
f_eda.afficher_cat_vars_univarie_tableau(df, 'BAD')
f_eda.afficher_num_vars_univarie_graph(df, palette=["#3336ff"])
df.columns

# Observations de valeurs extrêmes 
df[df['CLAGE']>650]
len(df[df['DEBTINC']>55])
len(df[df['DEBTINC']>100])

# Bivarie
ma_palette = ["#ff0000", "#3511ca"]

f_eda.afficher_cat_vars_bivarie_graph(df, 'BAD', palette = ma_palette)
f_eda.afficher_num_vars_bivarie_graph(df, 'BAD', palette = ma_palette)
f_eda.afficher_cat_vars_bivarie_tableau(df, 'BAD')

# Observations de DEROG X BAD et de DELINQ X BAD
len(df[df['DEROG'] == 0]) / len(df)
len(df[df['DELINQ'] == 0]) / len(df)
df_bad_0 = df[df['BAD'] == 0]

len(df[df['YOJ']==0.0]) / len(df)
len(df_bad_0[df_bad_0['YOJ']==0.0]) / len(df_bad_0)

len(df_bad_0[df_bad_0['DEROG'] == 0]) / len(df_bad_0)
len(df_bad_0[df_bad_0['DELINQ'] == 0]) / len(df_bad_0)

# correlation entre variables categorielles
f_eda.v_cramer_test_khi2(df['REASON'], df['JOB']) # association faible mais stats significative
f_eda.v_cramer_test_khi2(df['BAD'], df['JOB']) # association faible mais stats significative


# correlation entre 1 var cat x 1 var num
for var in vars_num:
    print('correlation_ratio entre ',var ,' et BAD: ', f_eda.correlation_ratio(df['BAD'], df[var]))


# Multivarie

f_eda.afficher_matrice_correlation_num(df, methode='spearman')
f_eda.afficher_matrice_correlation_num(df, methode='kendall')
f_eda.afficher_matrice_correlation_num(df)

# Derog et Delag corrélés à la variable y
# Value et Mortdue fortement et postiviement corrélés
sns.regplot(x=df['VALUE'], y=df['MORTDUE'], data=df, color = '#27006c')
plt.xlabel('VALUE')
plt.ylabel('MORTDUE')
plt.title('VALUE x MORTDUE')
plt.show()

# Prétraitement des données #######################################################################################

# Traitement des valeurs aberrantes par plafonnement

df.loc[df['CLAGE'] > 660, 'CLAGE'] = 660 # 55 ans
# df.loc[df['DEBTINC'] > 55, 'DEBTINC'] = 55


# Traitement des valeurs manquantes
f_pre.afficher_pourcentage_valeurs_manquantes(df)
f_pre.afficher_valeurs_manquantes_barplot(df)

# Supression des lignes avec un nb de valeurs manquantes > 5
tab_nb_va_manquantes = f_pre.afficher_nb_valeurs_manquantes(df)
# lignes avec au moins nb_vm valeurs manquante
for nb_vm in range(len(df.columns)):
    pourcent_vm = len(tab_nb_va_manquantes[tab_nb_va_manquantes > nb_vm]) / len(df)
    print('Plus de ',nb_vm, 'valeurs manquantes pour ' ,round(100 * pourcent_vm,2)," % des lignes")

# Observons ces lignes avec de nombreuses valeurs manquantes
index_a_enlever = tab_nb_va_manquantes[tab_nb_va_manquantes > 4].index
df_vm_4 = df.loc[index_a_enlever]
f_eda.afficher_cat_vars_univarie_graph(df_vm_4, 'BAD', palette=["blue"])
f_eda.afficher_num_vars_univarie_graph(df_vm_4, palette=["#3336ff"])


df = df.drop(index=index_a_enlever)
df = df.reset_index(drop=True)

# Imputation
df["REASON"].fillna("Other", inplace=True)
df["JOB"].fillna('Other',inplace=True)

# imputation par la mediane
for var in ["MORTDUE", "YOJ", "NINQ", "CLAGE", "CLNO", "VALUE"]:
    df[var] = df[var].fillna(value=df[var].median())

# imputation par 0 pour DEROG et DELINQ
# Observation préliminaire : DEROG NaN vs O
df_DEROG_0 = df[df['DEROG'] == 0]
df_DEROG_na = df[df['DEROG'].isna()]

len(df_DEROG_0[df_DEROG_0['BAD'] == 1]) / len(df_DEROG_0)
len(df_DEROG_na[df_DEROG_na['BAD'] == 1]) / len(df_DEROG_na)

# Observation préliminaire : DELINQ NaN vs O
df_DELINQ_0 = df[df['DELINQ'] == 0]
df_DELINQ_na = df[df['DELINQ'].isna()]
len(df_DELINQ_0[df_DELINQ_0['BAD'] == 1]) / len(df_DELINQ_0)
len(df_DELINQ_na[df_DELINQ_na['BAD'] == 1]) / len(df_DELINQ_na)

for var in ["DEROG", "DELINQ"]:
    df[var] = df[var].fillna(value=0)

# imputation de DEBTINC

# test de plusieurs méthodes
# observation des valeurs manquantes pour cette variables
df_DEBTINC_na = df[df['DEBTINC'].isna()]
len(df_DEBTINC_na[df_DEBTINC_na['BAD'] == 1]) / len(df_DEBTINC_na)
len(df[df['BAD'] == 1]) / len(df)

# test d'imputation sur DEBTINC
df_impute = f_pre.random_forest_imputation(df, 'DEBTINC')
print('imputation random forest : ')
print('df précédent -> correlation_ratio entre DEBTINC et BAD: ', f_eda.correlation_ratio(df['BAD'], df['DEBTINC']))
print('df imputé -> correlation_ratio entre DEBTINC et BAD: ', f_eda.correlation_ratio(df_impute['BAD'], df_impute['DEBTINC']))
print('-'*100)
df_impute = df.copy()
df_impute['DEBTINC'] = df_impute['DEBTINC'].fillna(value=df_impute['DEBTINC'].median())
print('imputation mediane : ')
print('df précédent -> correlation_ratio entre DEBTINC et BAD: ', f_eda.correlation_ratio(df['BAD'], df['DEBTINC']))
print('df imputé -> correlation_ratio entre DEBTINC et BAD: ', f_eda.correlation_ratio(df_impute['BAD'], df_impute['DEBTINC']))
print('-'*100)
df_impute = df.copy()
df_impute['DEBTINC'] = df_impute['DEBTINC'].fillna(value=df_impute['DEBTINC'].mean())
print('imputation moyenne : ')
print('df précédent -> correlation_ratio entre DEBTINC et BAD: ', f_eda.correlation_ratio(df['BAD'], df['DEBTINC']))
print('df imputé -> correlation_ratio entre DEBTINC et BAD: ', f_eda.correlation_ratio(df_impute['BAD'], df_impute['DEBTINC']))
print('-'*100)

df_impute = df.copy()
df_impute['DEBTINC'] = df_impute['DEBTINC'].fillna(value=0)
print('imputation par 0 : ')
print('df précédent -> correlation_ratio entre DEBTINC et BAD: ', f_eda.correlation_ratio(df['BAD'], df['DEBTINC']))
print('df imputé -> correlation_ratio entre DEBTINC et BAD: ', f_eda.correlation_ratio(df_impute['BAD'], df_impute['DEBTINC']))
print('-'*100)

df_impute = df.copy()
df_impute['DEBTINC'] = df_impute['DEBTINC'].fillna(value=55)
print('imputation par 55 : ')
print('df précédent -> correlation_ratio entre DEBTINC et BAD: ', f_eda.correlation_ratio(df['BAD'], df['DEBTINC']))
print('df imputé -> correlation_ratio entre DEBTINC et BAD: ', f_eda.correlation_ratio(df_impute['BAD'], df_impute['DEBTINC']))
print('-'*100)

ma_palette = ['#fd5d5d','#9afd5d']
f_eda.afficher_num_vars_bivarie_graph(df_impute, 'BAD', palette = ma_palette)


f_pre.afficher_pourcentage_valeurs_manquantes(df_impute) # verification



# Feature Engineering

# nouvelles variables
df_impute['LOAN_VALUE_RATIO'] = df_impute['LOAN'] / df_impute['VALUE']
df_impute['MORTDUE_VALUE_RATIO'] = df_impute['MORTDUE'] / df_impute['VALUE']
df_impute['CREDIT_PER_MONTH'] = df_impute['CLNO'] / df_impute['CLAGE']

# gestion des ['CLAGE'] == 0
mask_inf = np.isinf(df_impute['CREDIT_PER_MONTH'])
df_impute.loc[mask_inf, 'CREDIT_PER_MONTH'] = df_impute['CLNO']

for var in ['LOAN_VALUE_RATIO', 'MORTDUE_VALUE_RATIO', 'CREDIT_PER_MONTH']:
    print('correlation_ratio entre ',var ,' et BAD: ', f_eda.correlation_ratio(df_impute['BAD'], df_impute[var]))


colonnes_pour_interaction = ["DEROG", "DELINQ","DEBTINC"]
df_modelisation = f_pre.creation_variables_interaction(df_impute, cols = colonnes_pour_interaction)
df_modelisation.info()
len(df.columns)
len(df_modelisation.columns)

# Encodage
df_encoded = pd.get_dummies(df_modelisation, drop_first=True)
df_encoded.columns

# Normalisation / Standardisation

df_normalise = f_pre.normaliser_variables(df_encoded, 'BAD', scaler = "standard")
f_eda.afficher_df_head(df_normalise)

# transformation des variables booléennes en 0 et 1
df_clean = df_normalise.applymap(lambda x: int(x) if isinstance(x, bool) else x)
f_eda.afficher_df_head(df_clean)
df_clean.info()

# Modélisation #######################################################################################

# Choix des variables
vars_selectionne = f_m.stepwise_selection(df_clean, 'BAD')
vars_selectionne

# Test statistiques
f_ts.test_homoscedasticite(df_clean[vars_selectionne], df_clean['BAD'])
dw_stat = f_ts.test_independance_erreurs(df_clean[vars_selectionne], df_clean['BAD'])
vif_df= f_ts.test_independance_variables(df_clean[vars_selectionne])
vif_df

vars_selectionne.remove('DEBTINC**2')
# vars_selectionne.append('DEBTINC**2')
vif_df= f_ts.test_independance_variables(df_clean[vars_selectionne])
vif_df

# Summary d'une reg log

summary_reg_log = f_m.regression_logistique_simple_summary(df_clean, vars_selectionne, var_y ='BAD')
print(summary_reg_log)

# Analyse points influents

summary_p_influents = f_m.detecte_points_influents(df_clean, vars_selectionne, var_y ='BAD', seuil_cook_d = 0.01)
print(summary_p_influents)

# Fine-tuning d'une reg log
best_model, best_params = f_m.regression_logistique_kfold_gridsearch(df_clean, vars_selectionne, 'BAD')

X = df_clean[vars_selectionne]
y = df_clean['BAD']
y_pred_proba = best_model.predict_proba(X)[:, 1]
y_pred_proba = y_pred_proba.reshape(-1, 1)


f_m.plot_courbe_roc(y, y_pred_proba, color='blue', title= 'Courbe ROC et AUC pour le modèle de régression logistique')

# Choix du meilleur modèle
resultats_test_modeles = f_m.tester_modeles(df_clean,  vars_selectionne, target_variable = 'BAD')

# Fine-tuning d'une random forest
best_model, best_params = f_m.random_forest_kfold_gridsearch(df_clean, vars_selectionne, 'BAD')

y_pred_proba = best_model.predict_proba(X)[:, 1]

f_m.plot_courbe_roc(y, y_pred_proba, color='red', title= 'Courbe ROC et AUC pour le modèle de Random Forest')

# Clustering #######################################################################################

X = df_clean[vars_selectionne]
y_pred_proba = best_model.predict_proba(X)[:, 1]
y_pred_proba = y_pred_proba.reshape(-1, 1)


f_m.elbow_method(y_pred_proba)
km, labels, centroids = f_m.K_means(y_pred_proba, 3)

# ajout des clusters du Kmeans à mon df
df_clean['Classe_de_risque'] = labels
centroids
cluster_mapping = {0: 'AAA', 1: 'CCC', 2: 'BBB'}
df_clean['Classe_de_risque'] = df_clean['Classe_de_risque'].map(cluster_mapping)

# BAD X Classe de risque
vars_a_traiter = vars_selectionne + ['BAD']

for var in vars_a_traiter:
    print(var)
    df_clean.groupby('Classe_de_risque')[var].mean()
    print()

