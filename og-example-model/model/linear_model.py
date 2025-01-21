import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

class RidgePredictor:
    def __init__(self):
        self.model = None
        self.cv_model = GridSearchCV(
            Ridge(random_state=42, max_iter=2000, tol=1e-4, solver='svd'),
            param_grid={'alpha': np.logspace(-6, 2, 30)},
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        self.feature_importance = None
    
    def fit(self, X, y, feature_names=None):
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        self.cv_model.fit(X, y)
        self.model = self.cv_model.best_estimator_
        
        if feature_names is not None:
            importance = np.abs(self.model.coef_)
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        return self
    
    def predict(self, X):
        X = X.astype(np.float32)
        return self.model.predict(X).astype(np.float32)
    
    def get_best_params(self):
        return self.cv_model.best_params_
    
    def get_feature_importance(self, feature_names=None):
        if self.model is None:
            return None
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.model.coef_))]
        
        importance = np.abs(self.model.coef_)
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
