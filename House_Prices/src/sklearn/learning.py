import pandas as pd
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

with open('./db/submission/pre/data_pre_processed_train.pkl', mode='rb') as f:
    X_train_processed, X_train_processed_pca, Y_train = pk.load(f)

model_linear = LinearRegression()
model_tree = DecisionTreeRegressor(random_state=42)
model_random_forest = RandomForestRegressor(random_state=42)
model_svr = SVR(max_iter=10000)
model_neural_network = MLPRegressor(max_iter=10000, random_state=42)

cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)

linear_params = {
    'fit_intercept': [True, False],
    'copy_X': [True, False]
}

tree_params = {
    'max_depth': [None, 8, 12, 15, 20],
    'min_samples_split': [2, 5, 10, 20], 
    'min_samples_leaf': [1, 2, 4, 8], 
    'max_features': [1.0, 'sqrt', 'log2', 0.8], 
}

random_forest_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [1.0, 'sqrt', 0.7],
    'bootstrap': [True, False]
}

svr_params = {
    'kernel': ['rbf', 'linear'],
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.2, 0.5],
}

neural_network_params = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    'activation': ['relu', 'tanh'], 
    'solver': ['adam', 'sgd'], 
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate_init': [0.001, 0.01, 0.05],
    'learning_rate': ['constant', 'adaptive'],
}

train_models = {
    'linear': [model_linear, linear_params],
    'tree': [model_tree, tree_params],
    'random_forest': [model_random_forest, random_forest_params],
    'svr': [model_svr, svr_params],
    'neural_network': [model_neural_network, neural_network_params]
}

process = {
    'no_pca': X_train_processed,
    'pca': X_train_processed_pca
}

results = []

#Y_test_original = np.expm1(Y_test)

for n, (x) in process.items():
    for name, (model, param) in train_models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=param, verbose=2, cv=cv_strategy, n_jobs=-1, scoring='neg_root_mean_squared_error')
        grid_search.fit(x, Y_train)
        #prev = np.expm1(grid_search.predict(y))

        results.append({
            'type': n,
            'model_name': name,
            'model': grid_search.best_estimator_,
            'score': grid_search.best_score_,
            #'rmse': np.sqrt(mean_squared_error(Y_test_original, prev)),
            #'r2': r2_score(Y_test_original, prev),
            #'mae': mean_absolute_error(Y_test_original, prev)
        })
        
        with open(f'./db/submission/models/{n}_{name}_model.pkl', mode='wb') as f:
            pk.dump(grid_search.best_estimator_, f)

df = pd.DataFrame(results)
df.to_csv('./db/submission/datas/results.csv', sep=',', index=False)
