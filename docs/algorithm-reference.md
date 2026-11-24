# Algorithm reference

The package is `ml_from_scratch`. The main implementations are in `models.py`; metrics are in `metrics.py`. Each fitted model exposes its learned parameters and the relevant prediction/transform method. All examples use finite two-dimensional numerical feature matrices. Classification labels are numeric or orderable homogeneous scalar values; binary logistic uses 0 and 1.

| Model | Mathematical idea | API |
|---|---|---|
| Linear regression | Least squares using SVD-backed solver | LinearRegression().fit(X,y).predict(Xnew) |
| Logistic regression | Stable sigmoid, average cross-entropy gradient | LogisticRegression(rate=.1, steps=2000).fit(X,y).predict_proba(Xnew) |
| KNN | Euclidean or Manhattan distance and stable voting | KNN(k=5, metric='euclidean').fit(X,y).predict(Xnew) |
| Gaussian NB | Log prior plus Gaussian log densities | GaussianNB().fit(X,y).predict(Xnew) |
| Decision tree | Minimize weighted Gini or entropy impurity | DecisionTree(max_depth=5,min_leaf=1).fit(X,y).predict(Xnew) |
| Random forest | Bootstrap plus random features at splits and majority voting | RandomForest(n_trees=10,max_depth=5,seed=42).fit(X,y).predict(Xnew) |
| K-means | Lloyd iterations, random or k-means++ initialization | KMeans(k=3).fit(X).predict(Xnew) |
| PCA | Center, SVD, project and reconstruct | PCA(n_components=2).fit(X).transform(Xnew) |
| Gradient descent | Least-squares half-MSE with batch size control | gradient_descent(X,y,rate=.05,epochs=100,batch_size=16) |

Metrics: MAE, MSE, RMSE, R2, accuracy, precision, recall, F1, confusion matrix, binary log loss and inertia. Undefined binary precision, recall and F1 are defined as zero. For constant targets, R2 returns 1 for perfect predictions and 0 otherwise. Input validation and numerical edge cases are covered in the tests within their documented scope.

See [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html) and [NumPy linear algebra](https://numpy.org/doc/stable/reference/routines.linalg.html) for reference concepts. The links are standard documentation URLs, not independently availability-tested here. Nine notebooks show mathematical intuition and executable use cases. See the research limitations before comparing models.
