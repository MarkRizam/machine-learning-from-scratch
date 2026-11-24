import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as RefLinear, LogisticRegression as RefLogistic
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB as RefNB
from sklearn.decomposition import PCA as RefPCA
from ml_from_scratch.models import LinearRegression, LogisticRegression, KNN, GaussianNB, PCA, sigmoid


def data():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 3))
    y = (x[:,0] + x[:,1] + rng.normal(size=200) > 0).astype(int)
    return x, y


def test_linear():
    x, _ = data()
    y = x @ [1, 2, -3] + 4
    np.testing.assert_allclose(LinearRegression().fit(x,y).predict(x), RefLinear().fit(x,y).predict(x), atol=1e-10)


def test_rank_deficient():
    x = np.ones((5,2))
    np.testing.assert_allclose(LinearRegression().fit(x,np.arange(5)).predict(x), 2)


def test_logistic():
    x,y = data()
    m = LogisticRegression(steps=5000).fit(x[:150],y[:150])
    r = RefLogistic(C=np.inf, tol=1e-10).fit(x[:150],y[:150])
    np.testing.assert_allclose(m.predict_proba(x[150:]),r.predict_proba(x[150:]),atol=1e-4)
    assert m.loss_history[-1] < m.loss_history[0]


@pytest.mark.parametrize('metric', ['euclidean','manhattan'])
def test_knn(metric):
    x,y = data()
    np.testing.assert_array_equal(KNN(5,metric).fit(x[:150],y[:150]).predict(x[150:]),KNeighborsClassifier(n_neighbors=5,metric=metric).fit(x[:150],y[:150]).predict(x[150:]))


def test_nb():
    x,y = data()
    np.testing.assert_array_equal(GaussianNB().fit(x[:150],y[:150]).predict(x[150:]),RefNB().fit(x[:150],y[:150]).predict(x[150:]))


def test_pca():
    x,_ = data()
    m = PCA(2).fit(x[:150]); r = RefPCA(2).fit(x[:150])
    np.testing.assert_allclose(m.inverse_transform(m.transform(x[150:])),r.inverse_transform(r.transform(x[150:])),atol=1e-10)


def test_stable_sigmoid():
    np.testing.assert_allclose(sigmoid(np.array([-1000.,0.,1000.])),[0,.5,1])


@pytest.mark.parametrize('x', [[], [[float('nan')]], [1,2]])
def test_invalid_matrix(x):
    with pytest.raises(ValueError): LinearRegression().fit(x,[1])

from ml_from_scratch.models import DecisionTree, RandomForest, KMeans, gradient_descent
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans as RefKMeans

@pytest.mark.parametrize('criterion',['gini','entropy'])
def test_tree(criterion):
    x=np.arange(20).reshape(-1,1); y=(x[:,0]>9).astype(int)
    m=DecisionTree(2,criterion=criterion).fit(x,y)
    np.testing.assert_array_equal(m.predict(x),DecisionTreeClassifier(max_depth=2,criterion=criterion).fit(x,y).predict(x))


def test_forest():
    x,y=data()
    a=RandomForest(5,3).fit(x[:100],y[:100]).predict(x[100:])
    b=RandomForest(5,3).fit(x[:100],y[:100]).predict(x[100:])
    np.testing.assert_array_equal(a,b)
    assert np.mean(a==y[100:])>.6

@pytest.mark.parametrize('init',['random','k-means++'])
def test_kmeans(init):
    rng=np.random.default_rng(5)
    x=np.vstack([rng.normal(loc=loc,scale=.1,size=(30,2)) for loc in [-5,0,5]])
    m=KMeans(3,init=init,seed=42).fit(x)
    r=RefKMeans(3,n_init=10,random_state=42).fit(x)
    assert m.inertia>=r.inertia_-1e-8
    assert m.inertia<100 if init=='k-means++' else np.isfinite(m.inertia)

@pytest.mark.parametrize('batch',[None,1,16])
def test_gradient(batch):
    x,_=data(); y=x@[1,2,-3]+4
    w,h=gradient_descent(x,y,epochs=200,batch_size=batch)
    np.testing.assert_allclose(w,[4,1,2,-3],atol=.001)
    assert h[-1]<h[0]
