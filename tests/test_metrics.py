import numpy as np
import pytest
from sklearn import metrics as ref
from ml_from_scratch import metrics as m

@pytest.mark.parametrize('ours,theirs',[(m.mae,ref.mean_absolute_error),(m.mse,ref.mean_squared_error),(m.rmse,ref.root_mean_squared_error),(m.r2,ref.r2_score)])
def test_regression(ours,theirs):
    assert ours([1,2,5],[2,2,4])==pytest.approx(theirs([1,2,5],[2,2,4]))

@pytest.mark.parametrize('ours,theirs',[(m.accuracy,ref.accuracy_score),(m.precision,ref.precision_score),(m.recall,ref.recall_score),(m.f1,ref.f1_score)])
def test_classification(ours,theirs):
    assert ours([0,1,0,1,1],[1,1,0,0,1])==pytest.approx(theirs([0,1,0,1,1],[1,1,0,0,1]))


def test_matrix():
    np.testing.assert_array_equal(m.confusion_matrix([0,1,0,1],[0,1,1,1]),ref.confusion_matrix([0,1,0,1],[0,1,1,1]))


def test_log_loss():
    assert m.log_loss([0,1,1],[.1,.6,.9])==pytest.approx(ref.log_loss([0,1,1],[.1,.6,.9]))


def test_inertia():
    assert m.inertia([[0,0],[2,0]],[[1,0]],[0,0])==2


def test_edge_cases():
    assert m.precision([0],[0])==0
    assert m.recall([0],[0])==0
    assert m.f1([0],[0])==0
    assert m.r2([1,1],[1,1])==1
    assert m.r2([1,1],[2,2])==0
    with pytest.raises(ValueError): m.mse([],[])
    with pytest.raises(ValueError): m.log_loss([1],[1.1])
    with pytest.raises(ValueError): m.inertia([[0]],[[0]],[-1])
