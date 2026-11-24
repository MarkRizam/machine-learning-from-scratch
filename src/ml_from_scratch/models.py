"""Small educational NumPy estimators; no sklearn estimators used here."""
import numpy as np


def matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite 2D matrix')
    return x


def paired(x, y):
    x, y = matrix(x), np.asarray(y)
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError('Expected one target per row')
    return x, y


def classified(x, y):
    """Accept finite numeric or homogeneous string labels only."""
    x, y = paired(x, y)
    if y.dtype.kind in 'biuf':
        if not np.isfinite(y).all():
            raise ValueError('Expected finite class labels')
    elif y.dtype.kind not in 'US':
        raise ValueError('Expected numeric or homogeneous string class labels')
    return x, y


def integer(value, minimum):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= minimum


def design(x):
    x = matrix(x)
    return np.column_stack((np.ones(len(x)), x))


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


class LinearRegression:
    def fit(self, x, y):
        x, y = paired(x, y)
        y = y.astype(float)
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite target')
        self.weights = np.linalg.lstsq(design(x), y, rcond=None)[0]
        return self

    def predict(self, x):
        return design(x) @ self.weights


class LogisticRegression:
    def __init__(self, rate=0.1, steps=2000):
        if not np.isfinite(rate) or rate <= 0 or not integer(steps, 1):
            raise ValueError('Invalid optimizer configuration')
        self.rate, self.steps = rate, steps

    def fit(self, x, y):
        x, y = paired(x, y)
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError('Expected both binary labels 0 and 1')
        x = design(x)
        self.weights = np.zeros(x.shape[1])
        self.loss_history = []
        for _ in range(self.steps):
            z = x @ self.weights
            self.loss_history.append(float(np.mean(np.logaddexp(0, z) - y*z)))
            self.weights -= self.rate * (x.T @ (sigmoid(z)-y)) / len(x)
            if not np.isfinite(self.weights).all():
                raise FloatingPointError('Optimizer diverged')
        return self

    def predict_proba(self, x):
        p = sigmoid(design(x) @ self.weights)
        return np.column_stack((1-p, p))

    def predict(self, x, threshold=0.5):
        if not 0 <= threshold <= 1:
            raise ValueError('Invalid threshold')
        return (self.predict_proba(x)[:, 1] >= threshold).astype(int)


class KNN:
    def __init__(self, k=5, metric='euclidean'):
        if not integer(k, 1) or metric not in ('euclidean', 'manhattan'):
            raise ValueError('Invalid k or metric')
        self.k, self.metric = k, metric

    def fit(self, x, y):
        self.x, self.y = classified(x, y)
        self.x, self.y = self.x.copy(), self.y.copy()
        if self.k > len(self.x):
            raise ValueError('k exceeds training size')
        return self

    def predict(self, x):
        result = []
        x = matrix(x)
        if x.shape[1] != self.x.shape[1]:
            raise ValueError("Feature count differs")
        for row in x:
            delta = np.abs(self.x-row)
            d = (delta**2).sum(1) if self.metric == 'euclidean' else delta.sum(1)
            labels, counts = np.unique(self.y[np.argsort(d, kind='stable')[:self.k]], return_counts=True)
            result.append(labels[np.argmax(counts)])
        return np.asarray(result)


class GaussianNB:
    def fit(self, x, y):
        x, y = classified(x, y)
        self.classes, counts = np.unique(y, return_counts=True)
        self.priors = counts/len(y)
        self.means = np.array([x[y == c].mean(0) for c in self.classes])
        self.variances = np.array([x[y == c].var(0) for c in self.classes]) + max(1e-12, 1e-9*x.var(0).max())
        return self

    def predict(self, x):
        x = matrix(x)
        if x.shape[1] != self.means.shape[1]:
            raise ValueError("Feature count differs")
        logp = np.log(self.priors) - 0.5*np.sum(np.log(2*np.pi*self.variances) + (x[:, None, :]-self.means)**2/self.variances, axis=2)
        return self.classes[np.argmax(logp, axis=1)]


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, x):
        x = matrix(x)
        if len(x) < 2 or not integer(self.n_components, 1) or self.n_components > min(x.shape):
            raise ValueError('Invalid component count or sample size')
        self.mean = x.mean(0)
        _, s, vt = np.linalg.svd(x-self.mean, full_matrices=False)
        self.components = vt[:self.n_components]
        self.explained_variance = s[:self.n_components]**2/(len(x)-1)
        return self

    def transform(self, x):
        x = matrix(x)
        if x.shape[1] != len(self.mean):
            raise ValueError("Feature count differs")
        return (x-self.mean) @ self.components.T

    def inverse_transform(self, z):
        return matrix(z) @ self.components + self.mean


class DecisionTree:
    def __init__(self, max_depth=5, min_leaf=1, criterion='gini', max_features=None, seed=42):
        if not integer(max_depth, 0) or not integer(min_leaf, 1) or criterion not in ('gini','entropy') or (max_features is not None and not integer(max_features, 1)):
            raise ValueError('Invalid tree configuration')
        self.max_depth, self.min_leaf, self.criterion = max_depth, min_leaf, criterion
        self.max_features, self.seed = max_features, seed

    def fit(self, x, y):
        x,y = classified(x,y)
        self.n_features = x.shape[1]
        self.rng = np.random.default_rng(self.seed)
        if self.max_features is not None and not 1 <= self.max_features <= self.n_features:
            raise ValueError('Invalid max_features')
        self.root = self._grow(x,y,0)
        return self

    def _impurity(self, y):
        p = np.unique(y, return_counts=True)[1]/len(y)
        return 1-(p*p).sum() if self.criterion == 'gini' else -(p*np.log2(p)).sum()

    def _grow(self, x, y, depth):
        labels, counts = np.unique(y,return_counts=True)
        leaf = labels[np.argmax(counts)]
        if depth >= self.max_depth or len(labels)==1 or len(y)<2*self.min_leaf:
            return leaf
        best, split = self._impurity(y), None
        features = range(x.shape[1]) if self.max_features is None else sorted(self.rng.choice(x.shape[1], self.max_features, replace=False))
        for j in features:
            values = np.unique(x[:,j])
            for t in values[:-1]/2 + values[1:]/2:
                left = x[:,j] <= t
                n = int(left.sum())
                if n < self.min_leaf or len(y)-n < self.min_leaf:
                    continue
                score = (n*self._impurity(y[left])+(len(y)-n)*self._impurity(y[~left]))/len(y)
                if score < best-1e-12:
                    best, split = score, (j,t,left)
        if split is None:
            return leaf
        j,t,left = split
        return (j,t,self._grow(x[left],y[left],depth+1),self._grow(x[~left],y[~left],depth+1))

    def predict(self,x):
        x = matrix(x)
        if x.shape[1] != self.n_features:
            raise ValueError('Feature count differs')
        def visit(row, node):
            while isinstance(node,tuple):
                j,t,l,r = node
                node = l if row[j]<=t else r
            return node
        return np.asarray([visit(row,self.root) for row in x])


class RandomForest:
    def __init__(self,n_trees=10,max_depth=5,max_features=None,seed=42):
        if not integer(n_trees, 1) or not integer(max_depth, 0) or (max_features is not None and not integer(max_features, 1)):
            raise ValueError('Invalid tree count')
        self.n_trees,self.max_depth,self.max_features,self.seed = n_trees,max_depth,max_features,seed

    def fit(self,x,y):
        x,y = classified(x,y)
        rng = np.random.default_rng(self.seed)
        self.trees=[]
        for _ in range(self.n_trees):
            indices = rng.integers(0,len(x),size=len(x))
            features = self.max_features if self.max_features is not None else max(1,int(np.sqrt(x.shape[1])))
            self.trees.append(DecisionTree(self.max_depth,max_features=features,seed=int(rng.integers(2**32))).fit(x[indices],y[indices]))
        return self

    def predict(self,x):
        votes=np.stack([t.predict(x) for t in self.trees],axis=1)
        result=[]
        for row in votes:
            labels, counts=np.unique(row,return_counts=True)
            result.append(labels[np.argmax(counts)])
        return np.asarray(result)


class KMeans:
    def __init__(self,k=3,init='k-means++',steps=100,tol=1e-6,seed=42):
        if not integer(k, 1) or init not in ('random','k-means++') or not integer(steps, 1) or not np.isfinite(tol) or tol<0:
            raise ValueError('Invalid clustering configuration')
        self.k,self.init,self.steps,self.tol,self.seed=k,init,steps,tol,seed

    def fit(self,x):
        x=matrix(x)
        if self.k>len(x) or len(np.unique(x,axis=0))<self.k:
            raise ValueError('Not enough distinct points')
        rng=np.random.default_rng(self.seed)
        if self.init=='random':
            centers=x[rng.choice(len(x),self.k,replace=False)].copy()
        else:
            centers=[x[rng.integers(len(x))]]
            while len(centers)<self.k:
                d=((x[:,None,:]-np.array(centers))**2).sum(2).min(1)
                centers.append(x[rng.choice(len(x),p=d/d.sum())])
            centers=np.array(centers)
        for i in range(self.steps):
            d=((x[:,None,:]-centers)**2).sum(2)
            labels=d.argmin(1)
            new=centers.copy()
            for j in range(self.k):
                if np.any(labels==j): new[j]=x[labels==j].mean(0)
                else: new[j]=x[np.argmax(d.min(1))]
            shift=np.linalg.norm(new-centers)
            centers=new
            if shift<=self.tol: break
        self.centers=centers
        self.labels=self.predict(x)
        self.inertia=float(((x-centers[self.labels])**2).sum())
        self.n_iter=i+1
        return self

    def predict(self,x):
        x = matrix(x)
        if x.shape[1] != self.centers.shape[1]:
            raise ValueError("Feature count differs")
        return ((x[:,None,:]-self.centers)**2).sum(2).argmin(1)


def gradient_descent(x,y,rate=0.05,epochs=100,batch_size=None,seed=42):
    """Least-squares optimizer. Batch size 1 gives SGD; None gives full batch."""
    x,y=paired(x,y)
    y=y.astype(float)
    if rate<=0 or not np.isfinite(rate) or not integer(epochs, 1) or not np.isfinite(y).all():
        raise ValueError('Invalid optimizer inputs')
    x=design(x)
    batch_size=len(x) if batch_size is None else batch_size
    if not integer(batch_size, 1) or batch_size>len(x):
        raise ValueError('Invalid batch size')
    rng=np.random.default_rng(seed)
    w=np.zeros(x.shape[1]); history=[]
    for _ in range(epochs):
        indices=rng.permutation(len(x))
        for start in range(0,len(x),batch_size):
            ix=indices[start:start+batch_size]
            a,b=x[ix],y[ix]
            w-=rate*(a.T@(a@w-b))/len(ix)
        loss=float(np.mean((x@w-y)**2)/2)
        if not np.isfinite(loss): raise FloatingPointError('Optimizer diverged')
        history.append(loss)
    return w,history
