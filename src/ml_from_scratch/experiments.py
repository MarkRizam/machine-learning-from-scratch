"""AIR-001: deterministic synthetic experiments. Run python experiments.py."""
import json
import platform
from pathlib import Path
import numpy as np
import sklearn
from sklearn import linear_model, neighbors, naive_bayes, tree, ensemble, cluster, decomposition
from ml_from_scratch.models import LinearRegression, LogisticRegression, KNN, GaussianNB, DecisionTree, RandomForest, KMeans, PCA, gradient_descent, design
from ml_from_scratch.metrics import mse, accuracy, f1, log_loss

SEEDS = (11, 22, 33)


def run():
    records=[]
    def add(exp, seed, config, **values):
        records.append(dict(experiment=exp,seed=seed,configuration=config,**{k:float(v) for k,v in values.items()}))
    for seed in SEEDS:
        rng=np.random.default_rng(seed)
        x=rng.normal(size=(200,3))
        yr=4+x@[1,2,-3]+rng.normal(scale=.5,size=200)
        yc=(x[:,0]+x[:,1]+rng.normal(size=200)>0).astype(int)
        tr,va,te=slice(0,120),slice(120,160),slice(160,200)
        a=LinearRegression().fit(x[tr],yr[tr]); b=linear_model.LinearRegression().fit(x[tr],yr[tr])
        w,h=gradient_descent(x[tr],yr[tr],epochs=500)
        add('EXP-001',seed,'OLS vs GD',ours=mse(yr[te],a.predict(x[te])),reference=mse(yr[te],b.predict(x[te])),gradient=mse(yr[te],design(x[te])@w),coefficient_max_delta=np.max(np.abs(a.weights-np.r_[b.intercept_,b.coef_])))
        a=LogisticRegression(steps=5000).fit(x[tr],yc[tr]); b=linear_model.LogisticRegression(C=np.inf,tol=1e-10).fit(x[tr],yc[tr])
        add('EXP-002',seed,'logistic',ours=accuracy(yc[te],a.predict(x[te])),reference=accuracy(yc[te],b.predict(x[te])),ours_f1=f1(yc[te],a.predict(x[te])),reference_f1=f1(yc[te],b.predict(x[te])),ours_logloss=log_loss(yc[te],a.predict_proba(x[te])[:,1]),reference_logloss=log_loss(yc[te],b.predict_proba(x[te])[:,1]))
        a=GaussianNB().fit(x[tr],yc[tr]); b=naive_bayes.GaussianNB().fit(x[tr],yc[tr])
        add('EXP-002',seed,'Gaussian NB',ours=accuracy(yc[te],a.predict(x[te])),reference=accuracy(yc[te],b.predict(x[te])))
        for k in (1,5,9):
            for metric in ('euclidean','manhattan'):
                a=KNN(k,metric).fit(x[tr],yc[tr]); b=neighbors.KNeighborsClassifier(n_neighbors=k,metric=metric).fit(x[tr],yc[tr])
                add('EXP-003',seed,f'k={k}, {metric}',ours=accuracy(yc[va],a.predict(x[va])),reference=accuracy(yc[va],b.predict(x[va])))
        for depth in (1,3,5):
            a=DecisionTree(depth).fit(x[tr],yc[tr]); b=tree.DecisionTreeClassifier(max_depth=depth,random_state=seed).fit(x[tr],yc[tr])
            add('EXP-004',seed,f'depth={depth}',ours=accuracy(yc[va],a.predict(x[va])),reference=accuracy(yc[va],b.predict(x[va])),ours_train=accuracy(yc[tr],a.predict(x[tr])),reference_train=accuracy(yc[tr],b.predict(x[tr])))
        for count in (3,7):
            for features in (1,2):
                a=RandomForest(count,3,features,seed).fit(x[tr],yc[tr]); b=ensemble.RandomForestClassifier(n_estimators=count,max_depth=3,max_features=features,random_state=seed,n_jobs=1).fit(x[tr],yc[tr])
                add('EXP-005',seed,f'trees={count}, features={features}',ours=accuracy(yc[va],a.predict(x[va])),reference=accuracy(yc[va],b.predict(x[va])))
        blobs=np.vstack([rng.normal(loc=loc,scale=.2,size=(30,2)) for loc in (-5,0,5)])
        for init in ('random','k-means++'):
            a=KMeans(3,init=init,seed=seed).fit(blobs); b=cluster.KMeans(3,init=init,n_init=1,random_state=seed).fit(blobs)
            add('EXP-006',seed,init,ours=a.inertia,reference=b.inertia_,ours_iterations=a.n_iter,reference_iterations=b.n_iter_)
        for k in (1,2):
            a=PCA(k).fit(x[tr]); b=decomposition.PCA(k,svd_solver='full').fit(x[tr])
            add('EXP-007',seed,f'components={k}',ours=mse(x[te].ravel(),a.inverse_transform(a.transform(x[te])).ravel()),reference=mse(x[te].ravel(),b.inverse_transform(b.transform(x[te])).ravel()),ours_variance=a.explained_variance.sum(),reference_variance=b.explained_variance_.sum())
        baseline=LinearRegression().fit(x[tr],yr[tr]); optimum=mse(yr[tr],baseline.predict(x[tr]))/2
        for batch in (1,16,120):
            for rate in (.01,.05):
                w,h=gradient_descent(x[tr],yr[tr],rate=rate,epochs=100,batch_size=batch,seed=seed)
                add('EXP-008',seed,f'batch={batch}, rate={rate}',ours=h[-1],reference=optimum,initial=h[0])
    return records


def main():
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rcParams['svg.hashsalt'] = 'AIR-001'
    import matplotlib.pyplot as plt
    root=Path.cwd()
    (root/'results').mkdir(exist_ok=True); (root/'figures').mkdir(exist_ok=True)
    records=run(); repeated=run()
    assert len(records)==len(repeated)
    for a,b in zip(records,repeated):
        for k in a:
            if isinstance(a[k],float): np.testing.assert_allclose(a[k],b[k],rtol=0,atol=1e-10)
            else: assert a[k]==b[k]
    payload={'project':'AIR-001','seeds':list(SEEDS),'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'platform':platform.system(),'machine':platform.machine()},'reproducibility':{'runs':2,'atol':1e-10,'rtol':0,'status':'pass'},'records':records}
    (root/'results'/'benchmarks.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    for exp in sorted(set(r['experiment'] for r in records)):
        rows=[r for r in records if r['experiment']==exp]
        configs=list(dict.fromkeys(r['configuration'] for r in rows))
        fig,ax=plt.subplots(figsize=(9,4.5))
        for i,key in enumerate(('ours','reference')):
            means=[np.mean([r[key] for r in rows if r['configuration']==c]) for c in configs]
            std=[np.std([r[key] for r in rows if r['configuration']==c],ddof=1) for c in configs]
            ax.bar(np.arange(len(configs))+(i-.5)*.34,means,.34,yerr=std,capsize=3,label=key)
        ax.set_xticks(range(len(configs)),configs,rotation=25,ha='right')
        ax.set_ylabel({'EXP-001':'Test MSE','EXP-002':'Test accuracy','EXP-003':'Validation accuracy','EXP-004':'Validation accuracy','EXP-005':'Validation accuracy','EXP-006':'Training inertia','EXP-007':'Test reconstruction MSE','EXP-008':'Training half-MSE'}[exp])
        ax.set_title(exp+' — synthetic data, mean ± sample SD (3 seeds)'); ax.legend(); fig.tight_layout()
        for ext in ('png','svg'): fig.savefig(root/'figures'/f'{exp}.{ext}')
        plt.close(fig)
    print(f'{len(records)} records; 8 experiments; two runs matched at atol=1e-10')


if __name__=='__main__': main()
