import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from model.metrics import calculate_metrics
import pandas as pd
from model.linear_model import RidgePredictor

def train_linear_model(features, target, feature_cols, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = RidgePredictor()
    model.cv_model.n_jobs = 1
    
    cv_metrics = []
    feature_importance_folds = []
    
    for train_idx, val_idx in tscv.split(features):
        train_features = features[train_idx]
        val_features = features[val_idx]
        train_target = target[train_idx]
        val_target = target[val_idx]
        
        model.fit(train_features, train_target, feature_cols)
        importance = model.get_feature_importance(feature_cols)
        if importance is not None:
            feature_importance_folds.append(importance)
        
        val_pred = model.predict(val_features)
        val_metrics = calculate_metrics(val_target, val_pred)
        cv_metrics.append(val_metrics)
    
    avg_metrics = {}
    for metric in cv_metrics[0].keys():
        values = [fold_metrics[metric] for fold_metrics in cv_metrics]
        avg_metrics[metric] = (np.mean(values), np.std(values))
    
    if feature_importance_folds:
        avg_importance = pd.concat(feature_importance_folds).groupby('feature').mean()
        print("\nTop 10 most important features:")
        print(avg_importance.sort_values('importance', ascending=False).head(10))
    
    model.fit(features, target, feature_cols)
    
    model_info = {
        'model_type': 'Ridge',
        'alpha': model.cv_model.best_params_['alpha'],
        'n_splits': n_splits
    }
    
    print("\nModel Performance Metrics:")
    for metric, (mean, std) in avg_metrics.items():
        if metric == 'MSE':
            print(f"- {metric}: {mean:.8f} (±{std:.8f})")
        else:
            print(f"- {metric}: {mean:.4f} (±{std:.4f})")
    
    return model, avg_metrics, model_info
    