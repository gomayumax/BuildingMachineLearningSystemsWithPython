import numpy as np
from scipy import sparse
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.cross_validation import KFold
from pprint import pprint

data = np.array([[int(tok) for tok in line.split('\t')[:3]] for line in open('data/ml-100k/u.data')])
ij = data[:,:2]
ij -= 1 # original data is in 1-based system
values = data[:,2]
reviews = sparse.csc_matrix((values,ij.T)).astype(float)

reg = ElasticNetCV(fit_intercept=True, alphas=[0.0125, 0.025,0.05,.125,.25,.5,1.,2.,4.])

def movie_norm(xc):
    xc = xc.copy().toarray()
    # 値が0でない要素の配列を作る
    x1 = np.array([xi[xi > 0].mean() for xi in xc])
    # NANを0に変換
    x1 = np.nan_to_num(x1)

    # 評価の平均値を引いて正規化する
    for i in range(xc.shape[0]):
        xc[i] -= (xc[i] > 0) * x1[i]
    # 正規化した配列とその平均値を返す
    return xc, x1

def learn_for(i):
    # 対象ユーザを取得
    u = reviews[i]
    # 対象ユーザ意外のユーザindexを取得
    us = np.delete(np.arange(reviews.shape[0]), i)
    # ほとんどゼロの行列を通常の行列に変換
    # 更に1次元配列に変換
    # 値が0より大きい要素のindexを取得
    ps, = np.where(u.toarray().ravel() > 0)
    x = reviews[us][:,ps].T
    y = u.data
    pprint(y)
    exit()
    err = 0
    eb = 0
    kf = KFold(len(y), n_folds=2)
    for train,test in kf:
        # 映画ごとに正規化を行う
        xc,x1 = movie_norm(x[train])
        reg.fit(xc, y[train]-x1)
        # テストの時も正規化
        xc,x1 = movie_norm(x[test])
        # 回帰直線上の値
        p = np.array([reg.predict(xi) for xi in  xc]).ravel()
        # 実データとの距離
        e = (p+x1)-y[test]
        # ２乗誤差
        err += np.sum(e*e)
        eb += np.sum( (y[train].mean() - y[test])**2 )
    return np.sqrt(err/float(len(y))), np.sqrt(eb/float(len(y)))

whole_data = []
for i in range(reviews.shape[0]):
    s = learn_for(i)
    print(s[0] < s[1])
    print(s)
    whole_data.append(s)
