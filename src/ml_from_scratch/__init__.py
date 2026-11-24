"""AIR-001 educational machine learning implementations."""
from .models import (LinearRegression, LogisticRegression, KNN, GaussianNB,
                     DecisionTree, RandomForest, KMeans, PCA, gradient_descent)

__version__ = '0.1.0'
__all__ = ['LinearRegression', 'LogisticRegression', 'KNN', 'GaussianNB',
           'DecisionTree', 'RandomForest', 'KMeans', 'PCA', 'gradient_descent']
