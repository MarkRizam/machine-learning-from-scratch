"""Generate nine standalone educational notebook sources."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
COMMON = '''import numpy as np
import matplotlib.pyplot as plt
from ml_from_scratch import models as m, metrics as metrics
rng = np.random.default_rng(42)
x = rng.normal(size=(160, 2))
y = (x[:, 0] + x[:, 1] + rng.normal(size=160) > 0).astype(int)
train, holdout = slice(0,120), slice(120,160)
'''
SPECS = [
('01-linear-regression','Linear regression',
 'Fit a linear predictor and distinguish residuals from prediction errors.',
 'For a design matrix A with a leading column of ones, minimize ||Aw-y||². A least-squares solver avoids directly inverting the normal-equation matrix. For x=[0,1,2], y=[1,3,5], the intercept is 1 and slope is 2.',
 '''from sklearn.linear_model import LinearRegression as Ref
x=rng.uniform(-2,2,size=(160,1)); y=1+2*x[:,0]+rng.normal(scale=.4,size=160)
a=m.LinearRegression().fit(x[train],y[train]); b=Ref().fit(x[train],y[train])
print('Weights:',a.weights); print('Holdout MSE:',metrics.mse(y[holdout],a.predict(x[holdout])),metrics.mse(y[holdout],b.predict(x[holdout])))
order=np.argsort(x[holdout,0]); plt.scatter(x[holdout,0],y[holdout],label='holdout'); plt.plot(x[holdout,0][order],a.predict(x[holdout])[order],label='fit'); plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.show()''',
 'Increase the noise and measure error on new data.','Coefficients may be unstable under collinearity.','linear_model.html'),
('02-logistic-regression','Logistic regression',
 'Interpret probabilities and binary cross-entropy; inspect convergence.',
 'Sigmoid maps a linear score z to probability p=1/(1+exp(-z)). At z=0, p=0.5; at z=ln(3), p=0.75. The average cross-entropy gradient is Aᵀ(p-y)/n.',
 '''from sklearn.linear_model import LogisticRegression as Ref
a=m.LogisticRegression(steps=5000).fit(x[train],y[train]); b=Ref(C=np.inf,tol=1e-10).fit(x[train],y[train])
print('Accuracy:',metrics.accuracy(y[holdout],a.predict(x[holdout])),metrics.accuracy(y[holdout],b.predict(x[holdout])))
print('Log loss:',metrics.log_loss(y[holdout],a.predict_proba(x[holdout])[:,1])); plt.plot(a.loss_history); plt.xlabel('Iteration'); plt.ylabel('Training cross-entropy'); plt.show()''',
 'Change the classification threshold on a validation set and compute precision and recall.','Unregularized logistic regression may have no finite optimum on separable data.','linear_model.html#logistic-regression'),
('03-knn','K-nearest neighbors',
 'Understand distance metrics and voting.',
 'Euclidean distance is sqrt(sum((a-b)²)); Manhattan distance is sum(abs(a-b)). Between (0,0) and (3,4) they are 5 and 7. KNN votes among the nearest k training rows.',
 '''from sklearn.neighbors import KNeighborsClassifier
ks=[1,3,5,7,9]; ours=[]; refs=[]
for k in ks:
    a=m.KNN(k).fit(x[train],y[train]); b=KNeighborsClassifier(n_neighbors=k).fit(x[train],y[train]); ours.append(metrics.accuracy(y[holdout],a.predict(x[holdout]))); refs.append(metrics.accuracy(y[holdout],b.predict(x[holdout])))
print('Accuracy at k=5:',ours[2],refs[2]); plt.plot(ks,ours,'o-',label='ours'); plt.plot(ks,refs,'x--',label='reference'); plt.xlabel('k'); plt.ylabel('Holdout accuracy'); plt.legend(); plt.show()''',
 'Repeat using Manhattan distance.','Use validation rather than test data for selecting k; distance is sensitive to feature scales.','neighbors.html'),
('04-gaussian-naive-bayes','Gaussian Naive Bayes',
 'Combine class priors and conditional likelihoods in log space.',
 'Each class score is log prior plus the sum of feature log densities. For priors 0.6 and 0.4 and likelihoods 0.2 and 0.5, unnormalized posterior weights are 0.12 and 0.20; the second class wins.',
 '''from sklearn.naive_bayes import GaussianNB as Ref
a=m.GaussianNB().fit(x[train],y[train]); b=Ref().fit(x[train],y[train]); p=a.predict(x[holdout])
print('Accuracy:',metrics.accuracy(y[holdout],p),metrics.accuracy(y[holdout],b.predict(x[holdout])))
plt.imshow(metrics.confusion_matrix(y[holdout],p)); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.xticks([0,1]); plt.yticks([0,1]); plt.colorbar(); plt.show()''',
 'Duplicate a feature and inspect prediction changes.','The conditional independence and Gaussian assumptions may not hold.','naive_bayes.html'),
('05-decision-tree','Decision trees',
 'Understand impurity and recursive splitting.',
 'Gini impurity is 1-sum(p_c²). A balanced binary node has impurity 0.5; a pure node has impurity 0. Split selection seeks to reduce weighted child impurity.',
 '''from sklearn.tree import DecisionTreeClassifier
depths=[1,2,3,4,5]; train_acc=[]; held=[]; refs=[]
for d in depths:
    a=m.DecisionTree(d).fit(x[train],y[train]); b=DecisionTreeClassifier(max_depth=d,random_state=42).fit(x[train],y[train]); train_acc.append(metrics.accuracy(y[train],a.predict(x[train]))); held.append(metrics.accuracy(y[holdout],a.predict(x[holdout]))); refs.append(metrics.accuracy(y[holdout],b.predict(x[holdout])))
print('Depth 3 accuracy:',held[2],refs[2]); plt.plot(depths,train_acc,label='train'); plt.plot(depths,held,label='holdout'); plt.plot(depths,refs,label='reference holdout'); plt.xlabel('Depth'); plt.ylabel('Accuracy'); plt.legend(); plt.show()''',
 'Compare Gini and entropy on a validation split.','Greedy splitting may miss globally optimal trees; tied split choices can differ.','tree.html'),
('06-random-forest','Random forests',
 'Study bootstrap diversity and random feature selection.',
 'Train independent trees on bootstrap samples, with random candidate features at each split. Five trees voting [0,1,1,0,1] predict 1 by majority. The reference may average class probabilities instead.',
 '''from sklearn.ensemble import RandomForestClassifier
counts=[3,5,9]; ours=[]; refs=[]
for n in counts:
    a=m.RandomForest(n,3,1,42).fit(x[train],y[train]); b=RandomForestClassifier(n_estimators=n,max_depth=3,max_features=1,random_state=42).fit(x[train],y[train]); ours.append(metrics.accuracy(y[holdout],a.predict(x[holdout]))); refs.append(metrics.accuracy(y[holdout],b.predict(x[holdout])))
print('Accuracy at 9 trees:',ours[-1],refs[-1]); plt.plot(counts,ours,'o-',label='ours'); plt.plot(counts,refs,'x--',label='reference'); plt.xlabel('Trees'); plt.ylabel('Holdout accuracy'); plt.legend(); plt.show()''',
 'Repeat across multiple seeds and report variability.','RNG, split search, tie handling and voting differ from the reference.','ensemble.html'),
('07-kmeans','K-means clustering',
 'Understand centroid assignment, initialization and inertia.',
 'Alternate nearest-centroid assignment and mean updates. Points 0 and 2 assigned to one center yield center 1 and squared-distance sum 2. The objective is nonconvex.',
 '''from sklearn.cluster import KMeans as Ref
x=np.vstack([rng.normal(loc=c,scale=.2,size=(30,2)) for c in [-5,0,5]])
a=m.KMeans(3,seed=42).fit(x); b=Ref(3,n_init=1,random_state=42).fit(x)
print('Inertia:',a.inertia,b.inertia_); plt.scatter(x[:,0],x[:,1],c=a.labels); plt.scatter(a.centers[:,0],a.centers[:,1],marker='x',s=150,label='centroids'); plt.xlabel('Feature 1'); plt.ylabel('Feature 2'); plt.legend(); plt.show()''',
 'Compare random and k-means++ initialization across seeds.','Cluster IDs are arbitrary and lower inertia does not establish semantic validity.','clustering.html#k-means'),
('08-pca','Principal component analysis',
 'Understand variance-preserving projection and reconstruction.',
 'Center data and compute SVD. The top right singular vectors span directions of highest sample variance. For centered points (-1,0),(1,0), one component along the x-axis reconstructs exactly.',
 '''from sklearn.decomposition import PCA as Ref
x[:,1]=.8*x[:,0]+.3*x[:,1]
a=m.PCA(1).fit(x[train]); b=Ref(1).fit(x[train]); z=a.transform(x[holdout]); rec=a.inverse_transform(z); ref=b.inverse_transform(b.transform(x[holdout]))
print('Reconstruction MSE:',metrics.mse(x[holdout].ravel(),rec.ravel()),metrics.mse(x[holdout].ravel(),ref.ravel())); plt.scatter(x[holdout,0],x[holdout,1],label='original'); plt.scatter(rec[:,0],rec[:,1],label='reconstruction'); plt.xlabel('Feature 1'); plt.ylabel('Feature 2'); plt.legend(); plt.show()''',
 'Compare different scales and training-only standardization.','Component signs are arbitrary; directions may be unstable with near-tied eigenvalues.','decomposition.html#pca'),
('09-gradient-descent','Gradient descent',
 'Compare full-batch, stochastic and mini-batch optimization.',
 'For half-MSE, the gradient is Aᵀ(Aw-y)/n. For f(w)=w²/2 with w=2 and learning rate 0.1, one step gives w=1.8. Batch sizes change the number of updates per epoch.',
 '''from sklearn.linear_model import LinearRegression as Ref
y=4+x@[1.,-2.]+rng.normal(scale=.5,size=160)
ref=Ref().fit(x[train],y[train]); optimum=metrics.mse(y[train],ref.predict(x[train]))/2
for batch in [None,1,16]:
    w,h=m.gradient_descent(x[train],y[train],rate=.05,epochs=100,batch_size=batch); print('Batch',batch,'final loss',h[-1]); plt.plot(h,label=str(batch))
plt.axhline(optimum,linestyle='--',label='Reference OLS optimum'); plt.xlabel('Epoch'); plt.ylabel('Training half-MSE'); plt.legend(); plt.show()''',
 'Try different learning rates and report unstable runs.','Equal epochs do not mean equal computation; constant-step SGD may oscillate.','sgd.html')]


def main():
    for slug,title,objective,theory,code,exercise,limits,reference in SPECS:
        nb=nbf.v4.new_notebook(cells=[
            nbf.v4.new_markdown_cell(f'# {title}\n\n**AIR-001 — educational lab**\n\n## Learning objective\n{objective}\n\n## Mathematical idea and worked example\n{theory}\n\n## Implementation\nThe imported package implements the algorithm in NumPy; the reference is independent. All example data are synthetic.'),
            nbf.v4.new_code_cell(COMMON),nbf.v4.new_code_cell(code),
            nbf.v4.new_markdown_cell(f'## Interpretation\nCompare the displayed model outputs under the stated configuration. This small example is illustrative, not a general performance claim.\n\n## Exercise\n{exercise}\n\n## Limitations\n{limits}\n\n## Reference\nhttps://scikit-learn.org/stable/modules/{reference}\n\nFor the separate controlled experiments, see `research/protocol.md` and `results/benchmarks.json`.')],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
        for i,cell in enumerate(nb.cells): cell.id=f'{slug[:2]}-cell-{i}'
        nbf.write(nb,ROOT/'notebooks'/f'{slug}.ipynb')

if __name__=='__main__': main()
