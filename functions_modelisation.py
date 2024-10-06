"""
Projet Scoring 
M2 MoSEF
Louis LEBRETON
Dataset: hmeq.csv

Fonctions pour modélisation
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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


# Choix du modèle

def tester_modeles(df_norm, target_variable):
    X = df_norm.drop(columns=[target_variable])
    y = df_norm[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)
    
    # modeles à tester
    modeles = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        # 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        'Gradient Boosting': GradientBoostingClassifier()
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


# Fine-tuning d'une reg log
def regression_logistique_kfold_gridsearch(df, var_y, var_x, k_folds = 5):
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
    plt.title('Matrice de confusion')
    plt.xlabel('valeurs prédites')
    plt.ylabel('valeurs réelles')
    plt.show()

    return best_model, best_params