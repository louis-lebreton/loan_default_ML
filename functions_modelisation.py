"""
Projet Scoring 
M2 MoSEF
Louis LEBRETON
Dataset: hmeq.csv

Fonctions pour modélisation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc

# Choix des variables

def stepwise_selection(df, y, threshold_in=0.05, threshold_out=0.05, verbose=True):
    """
    Selectionne pas-à-pas une variable à ajouter dans le modèle et une autre à enlever du modèle
    à chaque itération
    
    """
    X = df.drop(columns=[y])
    y = df[y]

    # liste des variables à selectionner
    list_var = []
    changement = True

    while changement:
        changement = False
        # vars restantes
        vars_restantes = list(set(X.columns) - set(list_var))
        new_pval = pd.Series(index=vars_restantes, dtype=float) # pvalue des vars à tester

        for var_a_tester in vars_restantes:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[list_var + [var_a_tester]]))).fit(disp=0)
            new_pval[var_a_tester] = model.pvalues[var_a_tester]
        
        best_pval = new_pval.min() # choix de la pvalue la plus basse
        if best_pval < threshold_in:
            changement = True
            best_var = new_pval.idxmin()
            list_var.append(best_var) # ajout d'1 var
            if verbose:
                print(f'ajout de {best_var} avec p-valeur = {best_pval:.4}')

        # modele logistique
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[list_var]))).fit(disp=0)
    
        pvalues = model.pvalues.iloc[1:]  # on ignore la constante
        worst_pval = pvalues.max()  # choix de la pvalue la plus haute

        if worst_pval > threshold_out:
            changement = True
            worst_var = pvalues.idxmax()
            list_var.remove(worst_var) # retrait d'1 var
            if verbose:
                print(f'retrait de {worst_var} avec p-valeur = {worst_pval:.4}')

    if verbose:
        print("\n" + "-"*50 + "\n")
        print('nombre de variables dans le df: ', len(df.columns))
        print('nombre de variables selectionnées: ', len(list_var))

    return list_var



# regression logistique classique
def regression_logistique_simple_summary(df, vars_selectionne, var_y ='BAD'):
    """
    Summary d'une regression logistique classique
    Statsmodel plus adapté ici pour obtenir odds ratios etc
    """
    X = df[vars_selectionne]
    y = df[var_y]

    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit()

    params = result.params  # coeffs
    summary = round(result.conf_int(),2)  # IC
    summary['Odds Ratio'] = params.apply(lambda x: round(np.exp(x),2))  # odds ratios
    summary.columns = ['2.5%', '97.5%', 'Odds Ratio']
    summary['p-value'] = round(result.pvalues,3) # p-values

    return summary


def detecte_points_influents(df, vars_selectionne, var_y ='BAD', seuil_cook_d = 0.1):
    """
    Identifie les points influents avec les distances de Cook
    """
    X = df[vars_selectionne]
    y = df[var_y]

    X = sm.add_constant(X)

    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    influence = result.get_influence()
    
    # distances de cook
    cooks_d = influence.cooks_distance[0]
    
    points_influents = np.where(cooks_d > seuil_cook_d)[0]
    summary_points_influents = pd.DataFrame({
        'index': points_influents,
        'cooks_Distance': cooks_d[points_influents]
    })
    
    return summary_points_influents.sort_values(ascending = False, by = 'cooks_Distance')

# Fine-tuning d'une reg log
def regression_logistique_kfold_gridsearch(df, var_x, var_y, k_folds = 5):
    """
    Permet de fine tuner une regression logistique grâce à une cross validation (k-fold) gridsearch
    """
    X = df[var_x]
    y = df[var_y]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999, stratify=y)
    
    # hyperparamètres
    param_gridsearch = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.05, 0.5, 1],
        'solver': ['saga']
    }

    # modèle logistique
    log_reg = LogisticRegression(max_iter=1000)

    # gridSearch: k-fold cross-validation
    grid_search = GridSearchCV(log_reg, param_gridsearch, cv=k_folds, scoring='f1')
    grid_search.fit(X_train, y_train)

    # meilleurs hyperparamètres
    best_params = grid_search.best_params_
    print(f"Meilleurs hyperparamètres : {best_params}")

    # evaluer le modèle sur l'ensemble de test
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # rapport de classification
    print("\nRapport de classification sur l'ensemble de test :")
    print(classification_report(y_test, y_pred))

    print("accuracy score : ", accuracy_score(y_test,y_pred))
    print("precision score : ", precision_score(y_test, y_pred, average="macro"))
    print("recall score : ", recall_score(y_test, y_pred, average="macro"))
    print("f1 score : ", f1_score(y_test, y_pred, average="macro"))
    
    # matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
    plt.title('Matrice de confusion pour le modèle de régression logistique')
    plt.xlabel('valeurs prédites')
    plt.ylabel('valeurs réelles')
    plt.show()

    return best_model, best_params

# Choix du modèle

def tester_modeles(df_norm, selected_variables, target_variable):
    X = df_norm[selected_variables]
    y = df_norm[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)
    
    # modeles à tester
    modeles = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier(eval_metric='logloss')
        
    }
    
    resultats = {}
    
    for nom_model, modele in modeles.items():
        modele.fit(X_train, y_train)
        y_pred = modele.predict(X_test)
        
        # metriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        rappel = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        resultats[nom_model] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': rappel,
            'F1-Score': f1
        }
    
    resultats_df = pd.DataFrame(resultats).T
    return resultats_df


# Fine-tuning d'un Random Forest
def random_forest_kfold_gridsearch(df, var_x, var_y, k_folds=5):
    """
    Permet de fine tuner une random forest grâce à une cross validation (k-fold) gridsearch
    """
    X = df[var_x]
    y = df[var_y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999, stratify=y)

    # hyperparametres testés
    param_gridsearch = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=999)

    # gridSearch k-fold crossvalidation
    grid_search = GridSearchCV(rf, param_gridsearch, cv=k_folds, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)


    best_params = grid_search.best_params_
    print(f"Meilleurs hyperparamètres : {best_params}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # rapport de classification
    print("\nRapport de classification sur l'ensemble de test :")
    print(classification_report(y_test, y_pred))

    print("accuracy score : ", accuracy_score(y_test, y_pred))
    print("precision score : ", precision_score(y_test, y_pred, average="macro"))
    print("recall score : ", recall_score(y_test, y_pred, average="macro"))
    print("f1 score : ", f1_score(y_test, y_pred, average="macro"))

    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
    plt.title('Matrice de confusion pour le Random Forest')
    plt.xlabel('valeurs prédites')
    plt.ylabel('valeurs réelles')
    plt.show()

    return best_model, best_params

# Clustering : K-Means

def elbow_method(X):
    # elbow method
    inertia_list = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=999)
        kmeans.fit(X)
        inertia_list.append(kmeans.inertia_)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), inertia_list, marker='o')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie intra-cluster')
    plt.title('Méthode du coude')
    plt.grid(True)
    plt.show()

def K_means(X, k):
    
    kmeans = KMeans(n_clusters=k, random_state=999)
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    return kmeans, labels, centroids

def plot_courbe_roc(y, y_pred_proba, title:str, color:str):
    """
    fonction pour tracer la courbe ROC
    """
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # traçage
    plt.figure()
    plt.plot(fpr, tpr, color=color, lw=2, label=f'Courbe ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=2)
    plt.xlabel('taux de faux positifs')
    plt.ylabel('taux de vrais positifs')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()