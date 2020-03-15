import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb


# Compare Algorithms

def CompareModels(X,Y,models,n_splits,scoring):
    # prepare models
    '''models = []
    models.append(('LR', LinearRegression()))
    models.append(('SVR', SVR(kernel = 'rbf')))
    models.append(('KNN', DecisionTreeRegressor(random_state = seed)))
    models.append(('CART', RandomForestRegressor(n_estimators = 10, random_state = seed)))
    models.append(('NB', xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                    max_depth = 5, alpha = 10, n_estimators = 200)))'''
    # evaluate each model in turn
    results = []
    names = []
    best = ["model",-100,-100]
    for name, model in models:
        kfold = KFold(n_splits=n_splits, shuffle=True)
        cv_results = cross_val_score(model, X, Y.ravel(), cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        
        print(f'''In {n_splits} rounds, {name} model got average {scoring} score of {abs(cv_results.mean())} with standard deviation of {cv_results.std()}.''')
        if cv_results.mean() > best[1]:
            best[0]=model
            best[1]=cv_results.mean()
            best[2]=cv_results.std()
    
    print(f'''----------------------
    The best model was {best[0]} with {scoring} score of {abs(best[1])}, and a {scoring} standard deviation of {abs(best[2])}.''')

    # boxplot algorithm comparison
'''    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()'''

    