import ast
import inspect
import numpy as np
import pytest
from ml_from_scratch import models as m


def test_no_reference_library_in_algorithms():
    tree=ast.parse(inspect.getsource(m))
    imports=[a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    imports += [n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert all(not name.startswith('sklearn') for name in imports)


def test_logistic_gradient_finite_difference():
    a=m.design([[0.1,1.],[1.,2.],[-1.,0.5]])
    y=np.array([0.,1.,0.]); w=np.array([.2,-.1,.3]); eps=1e-6
    def loss(w):
        z=a@w
        return np.mean(np.logaddexp(0,z)-y*z)
    numerical=[]
    for i in range(len(w)):
        d=np.zeros_like(w); d[i]=eps
        numerical.append((loss(w+d)-loss(w-d))/(2*eps))
    np.testing.assert_allclose(a.T@(m.sigmoid(a@w)-y)/len(y),numerical,atol=1e-8)


def test_knn_tie_is_deterministic():
    assert m.KNN(2).fit([[0],[2]],[1,0]).predict([[1]])[0]==0


def test_bad_labels_and_parameters():
    with pytest.raises(ValueError): m.LogisticRegression().fit([[0],[1]],[0,2])
    with pytest.raises(ValueError): m.KNN(0)
    with pytest.raises(ValueError): m.KMeans(3).fit([[0],[0],[1]])
    with pytest.raises(ValueError): m.PCA(3).fit([[0,1],[1,2]])
    with pytest.raises(ValueError): m.gradient_descent([[0],[1]],[0,1],batch_size=3)


@pytest.mark.parametrize('model',[m.KNN(1),m.GaussianNB()])
def test_classification_feature_mismatch(model):
    model.fit([[0,1],[1,0]],[0,1])
    with pytest.raises(ValueError): model.predict([[1]])
